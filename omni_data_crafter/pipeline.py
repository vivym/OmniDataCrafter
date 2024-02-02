import queue
import threading
from dataclasses import dataclass

import lancedb
import pyarrow as pa
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm

from .app.schemas.task import TaskCompletion
from .ops.object_detectors import YOLOV8TRTEngine
from .ops.ocr import PaddlePaddleOCRV4TRTEngine
from .ops.optical_flow_estimators import RAFT
from .ops.video_decoders import VPFVideoDecoder
from .ops.clip import ClipVisionEncoder, ClipVisionEncoderTRTEngine
from .utils.cvt import rgb_to_hsv_nhwc_uint8


@dataclass
class FrameQueueItem:
    type: str
    task_id: str
    video_path: str
    result_queue: queue.Queue
    frames: torch.Tensor | None = None
    frame_idx: int | None = None
    fps: float | None = None
    total_frames: int | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class PostProcessQueueItem:
    type: str
    task_id: str
    video_path: str
    batch_idx: int
    shape: tuple[int, int]
    preds: torch.Tensor
    result_queue: queue.Queue


class Saver:
    def __init__(self, output_path: str):
        self.output_path = output_path

        self.db = lancedb.connect(output_path)

        table_names = self.db.table_names()

        if "clip" in table_names:
            self.clip_table = self.db.open_table("clip")
        else:
            schema = pa.schema([
                pa.field("video_id", pa.utf8()),
                pa.field("frame_idx", pa.int32()),
                pa.field("clip_feature", pa.list_(pa.float32(), list_size=768)),
            ])
            self.clip_table = self.db.create_table("clip", schema=schema)

        if "optical_flow" in table_names:
            self.optical_flow_table = self.db.open_table("optical_flow")
        else:
            schema = pa.schema([
                pa.field("video_id", pa.utf8()),
                pa.field("frame_idx", pa.int32()),
                pa.field("optical_flow_score", pa.float32()),
            ])
            self.optical_flow_table = self.db.create_table("optical_flow", schema=schema)

        if "cut_scores" in table_names:
            self.cut_scores_table = self.db.open_table("cut_scores")
        else:
            schema = pa.schema([
                pa.field("video_id", pa.utf8()),
                pa.field("fps", pa.float32()),
                pa.field("cut_scores", pa.list_(pa.float32())),
            ])
            self.cut_scores_table = self.db.create_table("cut_scores", schema=schema)

        if "ocr" in table_names:
            self.ocr_table = self.db.open_table("ocr")
        else:
            schema = pa.schema([
                pa.field("video_id", pa.utf8()),
                pa.field("frame_idx", pa.int32()),
                pa.field("ocr_score", pa.float32()),
                pa.field("boxes", pa.list_(pa.list_(pa.list_(pa.int32, list_size=2), list_size=4))),
                pa.field("scores", pa.list_(pa.float32())),
            ])
            self.ocr_table = self.db.create_table("ocr", schema=schema)

    def put_clip_features(self, clip_features: torch.Tensor):
        ...

    def put_optical_flow(self, optical_flow: torch.Tensor):
        ...

    def put_cut_scores(self, cut_scores: torch.Tensor):
        ...

    def put_ocr_results(self, ocr_score: float, boxes, scores):
        ...

    def put_det_results(self):
        ...


class Pipeline:
    def __init__(
        self,
        batch_size: int,
        device_id: int = 0,
        raft_model_path: str = "./weights/raft_things.safetensors",
        ocr_model_path: str = "./weights/pp-ocr-v4-det-fp16.engine",
        det_model_path: str = "./weights/yolov8m-fp16.engine",
    ):
        self.batch_size = batch_size
        self.device_id = device_id
        device_id = 0
        self.device_str = f"cuda:{device_id}"
        self.device = torch.device(self.device_str)

        self.clip_encoder = ClipVisionEncoder(
            model_name="ViT-L/14",
            device=self.device,
        )

        self.clip_encoder = ClipVisionEncoderTRTEngine(
            "./weights/clip_vit_l14-fp16.engine",
            self.device,
        )

        state_dict = {}
        with safe_open(raft_model_path, framework="pt", device=self.device_str) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        raft = RAFT()
        raft.load_state_dict(state_dict)
        raft.eval()
        raft.to(self.device)
        self.raft = raft

        self.pp_ocr = PaddlePaddleOCRV4TRTEngine(ocr_model_path, self.device)
        self.yolo_v8 = YOLOV8TRTEngine(det_model_path, self.device)

        self.frame_queue: queue.Queue[FrameQueueItem | None] = queue.Queue(maxsize=64)
        self.post_process_queue: queue.Queue[PostProcessQueueItem | None] = queue.Queue(maxsize=64)

        self.process_thread = threading.Thread(target=self.process_thread_fn)
        self.post_process_thread = threading.Thread(target=self.post_process_thread_fn)

    def compute_cut_scores(
        self,
        frames: torch.Tensor,
        last_hsv_frame: torch.Tensor | None,
        last_hsv_frame_2fps: torch.Tensor | None,
        last_hsv_frame_8fps: torch.Tensor | None,
        start_2fps: float,
        start_8fps: float,
        stride_2fps: float,
        stride_8fps: float,
    ):
        hsv_frames = rgb_to_hsv_nhwc_uint8(frames)

        cur = start_2fps
        indices_2fps = []
        while round(cur) < frames.shape[0]:
            indices_2fps.append(round(cur))
            cur += stride_2fps
        start_2fps = cur - frames.shape[0]
        indices_2fps_tensor = torch.as_tensor(indices_2fps, dtype=torch.int64, device=frames.device)

        cur = start_8fps
        indices_8fps = []
        while round(cur) < frames.shape[0]:
            indices_8fps.append(round(cur))
            cur += stride_8fps
        start_8fps = cur - frames.shape[0]
        indices_8fps_tensor = torch.as_tensor(indices_8fps, dtype=torch.int64, device=frames.device)

        hsv_frames_2fps = hsv_frames[indices_2fps_tensor]
        hsv_frames_8fps = hsv_frames[indices_8fps_tensor]

        if last_hsv_frame is None:
            diff = (hsv_frames[:-1] - hsv_frames[1:]).abs().to(torch.float32)
        else:
            prev_hsv_frames = torch.cat([last_hsv_frame[None], hsv_frames[:-1]], dim=0)
            diff = (prev_hsv_frames - hsv_frames).abs().to(torch.float32)

        if hsv_frames_2fps.shape[0] > 0:
            if last_hsv_frame_2fps is None:
                diff_2fps = (hsv_frames_2fps[:-1] - hsv_frames_2fps[1:]).abs().to(torch.float32)
            else:
                prev_hsv_frames_2fps = torch.cat([
                    last_hsv_frame_2fps[None], hsv_frames_2fps[:-1]
                ], dim=0)
                diff_2fps = (prev_hsv_frames_2fps - hsv_frames_2fps).abs().to(torch.float32)

        if hsv_frames_8fps.shape[0] > 0:
            if last_hsv_frame_8fps is None:
                diff_8fps = (hsv_frames_8fps[:-1] - hsv_frames_8fps[1:]).abs().to(torch.float32)
            else:
                prev_hsv_frames_8fps = torch.cat([
                    last_hsv_frame_8fps[None], hsv_frames_8fps[:-1]
                ], dim=0)
                diff_8fps = (prev_hsv_frames_8fps - hsv_frames_8fps).abs().to(torch.float32)

        last_hsv_frame = hsv_frames[-1]
        cut_scores = diff.flatten(1, 2).mean(dim=1)

        if hsv_frames_2fps.shape[0] > 0:
            cut_scores_2fps = diff_2fps.flatten(1, 2).mean(dim=1)
            last_hsv_frame_2fps = hsv_frames_2fps[-1]
        else:
            cut_scores_2fps = []

        if hsv_frames_8fps.shape[0] > 0:
            cut_scores_8fps = diff_8fps.flatten(1, 2).mean(dim=1)
            last_hsv_frame_8fps = hsv_frames_8fps[-1]
        else:
            cut_scores_8fps = []

        return (
            cut_scores, cut_scores_2fps, cut_scores_8fps,
            last_hsv_frame, last_hsv_frame_2fps, last_hsv_frame_8fps,
            start_2fps, start_8fps,
            indices_2fps_tensor, indices_2fps, indices_8fps,
        )

    def apply_resize(self, images: torch.Tensor, size: int) -> torch.Tensor:
        height, width = images.shape[-2:]
        if height < width:
            resize_to = (size, round(width * size / height))
        else:
            resize_to = (round(height * size / width), size)

        return F.interpolate(images, size=resize_to, mode="bicubic")

    def apply_center_crop(self, images: torch.Tensor, factor: int = 32) -> torch.Tensor:
        height, width = images.shape[-2:]
        new_height = height // factor * factor
        new_width = width // factor * factor

        if new_height != height or new_width != width:
            start_h = (height - new_height) // 2
            end_h = start_h + new_height
            start_w = (width - new_width) // 2
            end_w = start_w + new_width
            images = images[..., start_h:end_h, start_w:end_w]

        return images

    @torch.no_grad()
    def process_thread_fn(self):
        while True:
            try:
                item = self.frame_queue.get(timeout=1)
                if item is None:
                    break
            except queue.Empty:
                continue

            try:
                if item.type == "start":
                    task_id = item.task_id
                    video_path = item.video_path
                    fps = item.fps
                    stride_2fps = fps / 2.0
                    stride_8fps = fps / 8.0
                    frames_2fps_det_list = []
                    total_frames_2fps = 0
                    last_hsv_frame = None
                    last_hsv_frame_2fps = None
                    last_hsv_frame_8fps = None
                    start_2fps = 0
                    start_8fps = 0
                    last_frame_2fps = None
                    batch_idx_det = 0
                    batch_idx_flow = 0
                    results = []
                elif item.type == "frames":
                    frames = item.frames
                    result_queue = item.result_queue
                    (
                        cut_scores, cut_scores_2fps, cut_scores_8fps,
                        last_hsv_frame, last_hsv_frame_2fps, last_hsv_frame_8fps,
                        start_2fps, start_8fps,
                        indices_2fps_tensor, indices_2fps, indices_8fps,
                    ) = self.compute_cut_scores(
                        frames, last_hsv_frame, last_hsv_frame_2fps, last_hsv_frame_8fps,
                        start_2fps, start_8fps, stride_2fps, stride_8fps,
                    )

                    results.append({
                        "type": "cut_scores",
                        "task_id": task_id,
                        "video_path": video_path,
                        "frame_idx": item.frame_idx,
                        "cut_scores": cut_scores,
                        "cut_scores_2fps": cut_scores_2fps,
                        "cut_scores_8fps": cut_scores_8fps,
                        "indices_2fps": indices_2fps,
                        "indices_8fps": indices_8fps,
                    })

                    # -> b, 3, h, w
                    frames = frames.permute(0, 3, 1, 2).float()
                    frames.div_(255.0)
                    frames.clamp_(0.0, 1.0)

                    frames_2fps = frames[indices_2fps_tensor]

                    if frames_2fps.shape[0] > 0:
                        frames_2fps_resized_clip = self.apply_resize(frames_2fps, self.clip_encoder.input_res)
                        height, width = frames.shape[-2:]
                        frames_2fps_resized_det = self.apply_resize(frames_2fps, min(height, width) // 2)

                        # clip
                        clip_features = self.clip_encoder.encode(frames_2fps_resized_clip)
                        clip_features = clip_features

                        results.append({
                            "type": "clip",
                            "task_id": task_id,
                            "video_path": video_path,
                            "frame_idx": item.frame_idx,
                            "clip_features": clip_features,
                        })

                        # center crop for det
                        frames_2fps_det = self.apply_center_crop(frames_2fps_resized_det, factor=32)

                        frames_2fps_det_list.append(frames_2fps_det)
                        total_frames_2fps += frames_2fps_det.shape[0]

                    # optical flow
                    if total_frames_2fps >= 64:
                        frames_2fps_det = torch.cat(frames_2fps_det_list, dim=0)
                        total_frames = frames_2fps_det.shape[0]

                        pp_ocr_preds = self.pp_ocr.detect(frames_2fps_det)
                        self.post_process_queue.put(
                            PostProcessQueueItem(
                                type="pp_orc",
                                task_id=task_id,
                                video_path=video_path,
                                batch_idx=batch_idx_det,
                                shape=tuple(frames_2fps_det.shape[-2:]),
                                preds=pp_ocr_preds,
                                result_queue=result_queue,
                            )
                        )

                        yolo_v8_preds = self.yolo_v8.detect(frames_2fps_det)
                        self.post_process_queue.put(
                            PostProcessQueueItem(
                                type="yolo_v8",
                                task_id=task_id,
                                video_path=video_path,
                                batch_idx=batch_idx_det,
                                shape=tuple(frames_2fps_det.shape[-2:]),
                                preds=yolo_v8_preds,
                                result_queue=result_queue,
                            )
                        )

                        batch_idx_det += 1

                        if last_frame_2fps is not None:
                            frames_2fps_det = torch.cat([last_frame_2fps[None], frames_2fps_det], dim=0)
                            offset = 1
                        else:
                            offset = 0

                        frames_2fps_flow = frames_2fps_det * 2 - 1
                        batch_size = 32
                        for i in range(0 + offset, total_frames + offset, batch_size):
                            if i + batch_size > total_frames + offset:
                                break

                            start = max(i - 1, 0)
                            end = min(i + batch_size, total_frames)

                            frames1 = frames_2fps_flow[start:end - 1]
                            frames2 = frames_2fps_flow[start + 1:end]

                            flows = self.raft(frames1, frames2, update_iters=12)

                            mag = torch.sqrt(flows[:, 0, ...] ** 2 + flows[:, 1, ...] ** 2)
                            optical_flow_scores = mag.flatten(1).mean(dim=1)
                            results.append({
                                "type": "optical_flow",
                                "task_id": task_id,
                                "video_path": video_path,
                                "batch_idx": batch_idx_flow,
                                "optical_flow_scores": optical_flow_scores,
                            })

                            batch_idx_flow += 1

                        last_frame_2fps = frames_2fps_det[-1]
                        frames_2fps_det_list = [frames_2fps_det[i:]]
                        total_frames_2fps = frames_2fps_det_list[-1].shape[0]
                elif item.type == "end":
                    # optical flow
                    if total_frames_2fps > 0:
                        frames_2fps_det = torch.cat(frames_2fps_det_list, dim=0)
                        total_frames = frames_2fps_det.shape[0]

                        pp_ocr_preds = self.pp_ocr.detect(frames_2fps_det)
                        self.post_process_queue.put(
                            PostProcessQueueItem(
                                type="pp_orc",
                                task_id=task_id,
                                video_path=video_path,
                                batch_idx=batch_idx_det,
                                shape=tuple(frames_2fps_det.shape[-2:]),
                                preds=pp_ocr_preds,
                                result_queue=result_queue,
                            )
                        )

                        yolo_v8_preds = self.yolo_v8.detect(frames_2fps_det)
                        self.post_process_queue.put(
                            PostProcessQueueItem(
                                type="yolo_v8",
                                task_id=task_id,
                                video_path=video_path,
                                batch_idx=batch_idx_det,
                                shape=tuple(frames_2fps_det.shape[-2:]),
                                preds=yolo_v8_preds,
                                result_queue=result_queue,
                            )
                        )

                        batch_idx_det += 1

                        if last_frame_2fps is not None:
                            frames_2fps_det = torch.cat([last_frame_2fps[None], frames_2fps_det], dim=0)
                            offset = 1
                        else:
                            offset = 0

                        frames_2fps_flow = frames_2fps_det * 2 - 1
                        batch_size = 32
                        if frames_2fps_det.shape[0] > 1:
                            for i in range(0 + offset, total_frames + offset, batch_size):
                                start = max(i - 1, 0)
                                end = min(i + batch_size, total_frames)

                                frames1 = frames_2fps_flow[start:end - 1]
                                frames2 = frames_2fps_flow[start + 1:end]

                                if frames1.shape[0] > 0 and frames2.shape[0] > 0:
                                    flows = self.raft(frames1, frames2, update_iters=12)

                                    mag = torch.sqrt(flows[:, 0, ...] ** 2 + flows[:, 1, ...] ** 2)
                                    optical_flow_scores = mag.flatten(1).mean(dim=1)

                                    results.append({
                                        "type": "optical_flow",
                                        "task_id": task_id,
                                        "video_path": video_path,
                                        "batch_idx": batch_idx_flow,
                                        "optical_flow_scores": optical_flow_scores,
                                    })

                                batch_idx_flow += 1

                        last_frame_2fps = None
                        frames_2fps_det_list = []
                        total_frames_2fps = 0

                    for res in results:
                        new_res = {}
                        for key, val in res.items():
                            if isinstance(val, torch.Tensor):
                                new_res[key] = val.cpu().tolist()
                            else:
                                new_res[key] = val
                        result_queue.put(new_res)

                    torch.cuda.empty_cache()
                    item.result_queue.put(
                        TaskCompletion(
                            id=task_id,
                            status="completed",
                            fps=item.fps,
                            total_frames=item.total_frames,
                            width=item.width,
                            height=item.height,
                        )
                    )
                else:
                    raise ValueError(f"unknown item type: {item.type}")
            except Exception as e:
                import io
                import traceback
                str_io = io.StringIO()
                traceback.print_exc(file=str_io)
                item.result_queue.put(
                    TaskCompletion(
                        id=task_id,
                        status="failed",
                        message=str_io.getvalue() + f"\nitem: {item}" + f"\n{frames_2fps_det.shape}",
                    )
                )

        print("process_thread_fn stopped.")

    def post_process_thread_fn(self):
        while True:
            try:
                item = self.post_process_queue.get(timeout=1)
                if item is None:
                    break
            except queue.Empty:
                continue

            if item.type == "yolo_v8":
                results = self.yolo_v8.post_process(item.preds.cpu().numpy())
                item.result_queue.put({
                    "type": "det",
                    "detector": "yolo_v8",
                    "task_id": item.task_id,
                    "batch_idx": item.batch_idx,
                    "shape": item.shape,
                    "results": results,
                })
            elif item.type == "pp_orc":
                boxes, scores, ocr_scores = self.pp_ocr.post_process(item.preds.cpu().numpy())
                item.result_queue.put({
                    "type": "ocr",
                    "detector": "pp_orc",
                    "task_id": item.task_id,
                    "batch_idx": item.batch_idx,
                    "shape": item.shape,
                    "boxes": boxes,
                    "scores": scores,
                    "ocr_scores": ocr_scores,
                })
            else:
                raise ValueError(f"unknown item type: {item.type}")

        print("post_process_thread_fn stopped.")

    def start(self):
        self.process_thread.start()
        self.post_process_thread.start()

    def close(self):
        self.frame_queue.put(None)
        self.post_process_queue.put(None)

        self.process_thread.join()
        self.post_process_thread.join()

    @torch.no_grad()
    def __call__(
        self,
        task_id: str,
        video_path: str,
        result_queue: queue.Queue,
        verbose: bool = False,
    ):
        print("video_path", video_path)

        decoder = VPFVideoDecoder(
            video_path=video_path,
            batch_size=self.batch_size,
            device_id=self.device_id,
        )

        if decoder.width != 1280 or decoder.height != 720:
            result_queue.put(
                TaskCompletion(
                    id=task_id,
                    status="failed",
                    message=(
                        "video resolution is not 720x1280 "
                        f"({decoder.height}x{decoder.width})."
                    ),
                )
            )
            return

        self.frame_queue.put(
            FrameQueueItem(
                type="start",
                task_id=task_id,
                video_path=video_path,
                fps=decoder.fps,
                result_queue=result_queue,
            ),
        )

        frame_idx = 0
        for frames in tqdm(decoder.iter_frames(pixel_format="rgb"), disable=not verbose):
            self.frame_queue.put(
                FrameQueueItem(
                    type="frames",
                    task_id=task_id,
                    video_path=video_path,
                    frames=frames,
                    frame_idx=frame_idx,
                    result_queue=result_queue,
                ),
            )
            frame_idx += frames.shape[0]

        self.frame_queue.put(
            FrameQueueItem(
                type="end",
                task_id=task_id,
                video_path=video_path,
                fps=decoder.fps,
                total_frames=decoder.total_frames,
                width=decoder.width,
                height=decoder.height,
                result_queue=result_queue,
            )
        )
