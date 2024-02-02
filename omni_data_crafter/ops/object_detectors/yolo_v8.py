import numpy as np
import torch
import onnxruntime
from ultralytics.utils import ops

from omni_data_crafter.utils.trt_engine import TRTEngine


class YOLOV8Onnx:
    def __init__(self, model_path: str):
        self.sess = onnxruntime.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

    def detect(self, images: torch.Tensor) -> np.ndarray:
        outputs = self.sess.run(None, {"images": images.cpu().numpy()})[0]
        return outputs

    def post_process(self, preds: np.ndarray):
        preds = torch.from_numpy(preds)
        preds = ops.non_max_suppression(
            preds,
            conf_thres=0.25,
            iou_thres=0.7,
            agnostic=False,
            max_det=300,
        )
        preds = [
            pred.cpu().tolist() if pred is not None else None
            for pred in preds
        ]
        return preds


class YOLOV8TRTEngine:
    def __init__(self, engine_path: str, device: torch.device):
        self.engine = TRTEngine(engine_path).load().activate()
        self.device = device

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)

    def detect(self, images: torch.Tensor) -> torch.Tensor:
        self.engine.allocate_buffers(
            shape_dict={
                "images": images.shape,
                "output0": (images.shape[0], 84, 4620),
            },
            device=self.device,
        )

        preds = self.engine.infer(
            {"images": images},
            torch.cuda.current_stream(self.device),
            use_cuda_graph=False,
        )["output0"]

        return preds

    def post_process(self, preds: np.ndarray):
        preds = torch.from_numpy(preds)
        preds = ops.non_max_suppression(
            preds,
            conf_thres=0.25,
            iou_thres=0.7,
            agnostic=False,
            max_det=300,
        )
        preds = [
            pred.cpu().tolist() if pred is not None else None
            for pred in preds
        ]
        return preds
