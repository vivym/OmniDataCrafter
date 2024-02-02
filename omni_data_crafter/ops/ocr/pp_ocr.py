import cv2
import numpy as np
import onnxruntime
import pyclipper
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon

from omni_data_crafter.utils.trt_engine import TRTEngine


class PaddlePaddleOCRV4TRTEngine:
    def __init__(self, engine_path: str, device: torch.device):
        self.engine = TRTEngine(engine_path).load().activate()
        self.device = device

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)

    def detect(self, images: torch.Tensor) -> torch.Tensor:
        # normalize
        images = (images - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        self.engine.allocate_buffers(
            shape_dict={
                "x": images.shape,
                "sigmoid_0.tmp_0": (images.shape[0], 1, images.shape[2], images.shape[3]),
            },
            device=self.device,
        )

        preds = self.engine.infer(
            {"x": images},
            torch.cuda.current_stream(self.device),
            use_cuda_graph=False,
        )["sigmoid_0.tmp_0"]

        return preds[:, 0, ...]

    def post_process(self, preds: np.ndarray):
        bitmaps = preds > 0.3

        all_boxes = []
        all_scores = []
        ocr_scores = []
        for pred, bitmap in zip(preds, bitmaps):
            height, width = bitmap.shape

            outs = cv2.findContours(
                bitmap.astype("uint8") * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if len(outs) == 3:
                _, contours, _ = outs
            else:
                contours, _ = outs

            boxes = []
            scores = []
            for contour in contours[:1000]:
                points, sside = get_mini_boxes(contour)
                if sside < 3:
                    continue
                points = np.array(points)
                score = box_score_fast(pred, points.reshape(-1, 2))
                if score < 0.6:
                    continue

                box = unclip(points, 1.5).reshape(-1, 1, 2)
                box, sside = get_mini_boxes(box)
                if sside < 5:
                    continue
                box = np.array(box)
                boxes.append(box.astype("int32"))
                scores.append(score)

            area = 0
            for box in boxes:
                area += cv2.contourArea(box)
            ocr_scores.append(area / (height * width))

            boxes = [box.tolist() for box in boxes]
            all_boxes.append(boxes)
            all_scores.append(scores)

        return all_boxes, all_scores, ocr_scores


class PaddlePaddleOCRV4Onnx:
    def __init__(self, model_path: str, device: torch.device):
        self.sess = onnxruntime.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)

    def detect(self, images: torch.Tensor):
        # orig_height, orig_width = images.shape[-2:]

        # if max(orig_height, orig_width) > 960:
        #     if orig_height > orig_width:
        #         resize_to = (960, round(orig_width * 960 / orig_height))
        #     else:
        #         resize_to = (round(orig_height * 960 / orig_width), 960)

        #     images = F.interpolate(images, resize_to, mode="bilinear")

        # new_height = images.shape[-2] // 32 * 32
        # top = (images.shape[-2] - new_height) // 2
        # bottom = top + new_height

        # new_width = images.shape[-1] // 32 * 32
        # left = (images.shape[-1] - new_width) // 2
        # right = left + new_width

        # images = images[..., top:bottom, left:right]


        # print("images", images.shape, orig_height, orig_width)

        # normalize
        images = (images - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        preds = self.sess.run(None, {"x": images.cpu().numpy()})[0][:, 0, :, :]
        print("preds", preds.shape)

        return preds

    def post_process(self, preds: np.ndarray):
        bitmaps = preds > 0.3

        all_boxes = []
        all_scores = []
        ocr_scores = []
        for pred, bitmap in zip(preds, bitmaps):
            height, width = bitmap.shape

            outs = cv2.findContours(
                bitmap.astype("uint8") * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if len(outs) == 3:
                _, contours, _ = outs
            else:
                contours, _ = outs

            boxes = []
            scores = []
            for contour in contours[:1000]:
                points, sside = get_mini_boxes(contour)
                if sside < 3:
                    continue
                points = np.array(points)
                score = box_score_fast(pred, points.reshape(-1, 2))
                if score < 0.6:
                    continue

                box = unclip(points, 1.5).reshape(-1, 1, 2)
                box, sside = get_mini_boxes(box)
                if sside < 5:
                    continue
                box = np.array(box)
                boxes.append(box.astype("int32").tolist())
                scores.append(score)

            all_boxes.append(boxes)
            all_scores.append(scores)
            area = 0
            for box in boxes:
                area += cv2.contourArea(box)
            ocr_scores.append(area / (height * width))

        return all_boxes, all_scores, ocr_scores


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])


def box_score_fast(bitmap, _box):
    """
    box_score_fast: use bbox mean score as the mean score
    """
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded
