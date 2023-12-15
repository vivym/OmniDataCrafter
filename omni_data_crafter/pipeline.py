import queue
import threading
from dataclasses import dataclass

import lancedb
import pyarrow as pa
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm

from .ops.ocr import PaddlePaddleOCRV4Onnx
from .ops.optical_flow_estimators import RAFT
from .ops.video_decoders import VPFVideoDecoder
from .ops.clip import ClipVisionEncoder
from .utils.cvt import rgb_to_hsv_nhwc_uint8


@dataclass
class FrameQueueItem:
    type: str
    frames: torch.Tensor | None = None
    frame_idx: int | None = None
    video_id: str | None = None
    fps: float | None = None


@dataclass
class SavingQueueItem:
    type: str
    video_id: str
    frame_idx: int
    feature: torch.Tensor


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


class Pipeline:
    def __init__(
        self,
        batch_size: int,
        device_id: int = 0,
        raft_model_path: str = "./weights/raft_things.safetensors",
        output_path: str = "./data/lancedb",
    ):
        self.batch_size = batch_size
        self.device_id = device_id
        self.device_str = f"cuda:{device_id}"
        self.device = torch.device(self.device_str)

        self.clip_encoder = ClipVisionEncoder(
            model_name="ViT-L/14",
            device=self.device,
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

        self.pp_ocr = PaddlePaddleOCRV4Onnx("./weights/pp-ocr-v4-det.onnx")

        self.frame_queue: queue.Queue[FrameQueueItem | None] = queue.Queue(maxsize=64)
        self.saving_queue: queue.Queue[SavingQueueItem | None] = queue.Queue(maxsize=1024)

        self.process_thread = threading.Thread(target=self.process_thread_fn)
        # self.saving_thread = threading.Thread(target=self.saving_thread_fn)

        self.saver = Saver(output_path)

    def process_thread_fn(self):
        frames_2fps_list = []
        total_frames_2fps = 0
        last_hsv_frames = None
        cut_scores_list = []
        start = 0
        fps = 30.0
        stride = 1
        video_id = ""

        while True:
            try:
                item = self.frame_queue.get(timeout=1)
                if item is None:
                    break
            except queue.Empty:
                continue

            if item.type == "start":
                video_id = item.video_id
                fps = item.fps
                stride = int(fps // 2)
                frames_2fps_list = []
                total_frames_2fps = 0
                last_hsv_frames = None
                cut_scores_list = []
                start = 0
            elif item.type == "frames":
                frames = item.frames
                hsv_frames = rgb_to_hsv_nhwc_uint8(frames)

                if last_hsv_frames is None:
                    diff = (frames[:-1] - frames[1:]).abs().to(torch.float32)
                else:
                    prev_hsv_frames = torch.cat([last_hsv_frames[-1:], hsv_frames[:-1]], dim=0)
                    diff = (prev_hsv_frames - hsv_frames).abs().to(torch.float32)
                last_hsv_frames = hsv_frames

                cut_scores = diff.flatten(1, 2).mean(dim=1).mean(dim=1)
                cut_scores_list.append(cut_scores)

                # -> b, 3, h, w
                frames = frames.permute(0, 3, 1, 2).float()
                frames.div_(255.0)
                frames.clamp_(0.0, 1.0)

                cur = start
                indices = []
                while cur < frames.shape[0]:
                    indices.append(cur)
                    cur += stride
                start = cur - frames.shape[0]
                indices = torch.as_tensor(indices, dtype=torch.int64, device=frames.device)

                frames_2fps = frames[indices]

                # resize
                height, width = frames_2fps.shape[-2:]
                input_res = self.clip_encoder.input_res
                assert input_res % 8 == 0
                if height < width:
                    resize_to = (input_res, round(width * input_res / height))
                else:
                    resize_to = (round(height * input_res / width), input_res)

                frames_2fps = F.interpolate(frames_2fps, size=resize_to, mode="bicubic")

                # clip
                clip_features = self.clip_encoder.encode(frames_2fps)
                clip_features = clip_features.cpu()

                # center crop
                height, width = frames_2fps.shape[-2:]
                if height > width:
                    new_height = height // 8 * 8
                    if new_height != height:
                        start = (height - new_height) // 2
                        end = start + new_height
                        frames_2fps = frames_2fps[..., start:end, :]
                else:
                    new_width = width // 8 * 8
                    if new_width != width:
                        start = (width - new_width) // 2
                        end = start + new_width
                        frames_2fps = frames_2fps[..., start:end]

                frames_2fps_list.append(frames_2fps)
                total_frames_2fps += frames_2fps.shape[0]

                # optical flow
                if total_frames_2fps >= 64:
                    frames_2fps = torch.cat(frames_2fps_list, dim=0)
                    total_frames = frames_2fps.shape[0]

                    boxes, scores, ocr_scores = self.pp_ocr.detect(frames_2fps)

                    batch_size = 32
                    for i in range(0, total_frames, batch_size):
                        if i + batch_size > total_frames:
                            break

                        start = max(i - 1, 0)
                        end = min(i + batch_size, total_frames)

                        frames1 = frames_2fps[start:end - 1]
                        frames2 = frames_2fps[start + 1:end]

                        flows = self.raft(frames1, frames2, update_iters=12)

                        mag = torch.sqrt(flows[..., 0] ** 2 + flows[..., 1] ** 2)
                        optical_flow_scores = mag.flatten(1).mean(dim=1).cpu()

                    frames_2fps_list = [frames_2fps[i:]]
                    total_frames_2fps = frames_2fps_list[-1].shape[0]
            elif item.type == "end":
                # optical flow
                if total_frames_2fps > 0:
                    frames_2fps = torch.cat(frames_2fps_list, dim=0)
                    total_frames = frames_2fps.shape[0]

                    batch_size = 32
                    for i in range(0, total_frames, batch_size):
                        start = max(i - 1, 0)
                        end = min(i + batch_size, total_frames)

                        frames1 = frames_2fps[start:end - 1]
                        frames2 = frames_2fps[start + 1:end]

                        flows = self.raft(frames1, frames2, update_iters=12)

                        mag = torch.sqrt(flows[..., 0] ** 2 + flows[..., 1] ** 2)
                        optical_flow_scores = mag.flatten(1).mean(dim=1).cpu()

                    frames_2fps_list = []
                    total_frames_2fps = 0
            else:
                raise ValueError(f"unknown item type: {item.type}")

        print("process_thread_fn stopped.")

    def saving_thread_fn(self):
        while True:
            try:
                item = self.saving_queue.get(timeout=1)
                if item is None:
                    break
            except queue.Empty:
                continue

            if item.type == "clip":
                self.saver.put_clip_features(item.feature)
            elif item.type == "optical_flow":
                self.saver.put_optical_flow(item.feature)
            elif item.type == "cut_scores":
                self.saver.put_cut_scores(item.feature)
            else:
                raise ValueError(f"unknown item type: {item.type}")

        print("saving_thread_fn stopped.")

    def start(self):
        self.process_thread.start()
        # self.saving_thread.start()

    def close(self):
        self.frame_queue.put(None)
        self.saving_queue.put(None)

        self.process_thread.join()
        # self.saving_thread.join()

    @torch.no_grad()
    def __call__(self, video_path: str):
        decoder = VPFVideoDecoder(
            video_path=video_path,
            batch_size=self.batch_size,
            device_id=self.device_id,
        )
        print("total_frames", decoder.total_frames)

        self.frame_queue.put(
            FrameQueueItem(
                type="start",
                video_id=video_path,
                fps=decoder.fps,
            ),
        )

        frame_idx = 0
        for frames in tqdm(decoder.iter_frames(pixel_format="rgb")):
            frames = frames.clone()
            # torch.cuda.synchronize()
            self.frame_queue.put(
                FrameQueueItem(
                    type="frames",
                    frames=frames,
                    frame_idx=frame_idx,
                ),
            )
            frame_idx += frames.shape[0]

        self.frame_queue.put(FrameQueueItem(type="end"))
