import clip
import torch
import torch.nn.functional as F

from omni_data_crafter.utils.trt_engine import TRTEngine


class ClipVisionEncoder:
    def __init__(self,
        model_name: str = "ViT-L/14",
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        self.clip_model, _ = clip.load(model_name, device=device, jit=False)
        self.input_res = self.clip_model.visual.input_resolution
        # self.input_res = self.clip_model.input_resolution.item()
        # self.clip_model.visual = torch.compile(self.clip_model.visual, fullgraph=True)

        # self.clip_model.half()

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        height, width = frames.shape[-2:]

        if min(height, width) != self.input_res:
            # resize
            if height < width:
                resize_to = (self.input_res, round(width * self.input_res / height))
            else:
                resize_to = (round(height * self.input_res / width), self.input_res)

            frames = F.interpolate(frames, resize_to, mode="bilinear")

        height, width = frames.shape[-2:]
        # center crop
        if height > width:
            top = (height - self.input_res) // 2
            bottom = top + self.input_res
            left = 0
            right = self.input_res
        else:
            top = 0
            bottom = self.input_res
            left = (width - self.input_res) // 2
            right = left + self.input_res

        frames = frames[..., top:bottom, left:right]

        # normalize
        frames = (frames - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        with torch.autocast(device_type="cuda"):
            features = self.clip_model.encode_image(frames)

            # l2 normalize
            return F.normalize(features, p=2, dim=-1)


class ClipVisionEncoderTRTEngine:
    def __init__(self, engine_path: str, device: torch.device):
        self.engine = TRTEngine(engine_path).load().activate()
        self.device = device

        self.input_res = 224

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        height, width = frames.shape[-2:]

        assert min(height, width) == 224, (height, width)

        # center crop
        if height > width:
            top = (height - 224) // 2
            bottom = top + 224
            left = 0
            right = 224
        else:
            top = 0
            bottom = 224
            left = (width - 224) // 2
            right = left + 224

        frames = frames[..., top:bottom, left:right]

        # normalize
        frames = (frames - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        self.engine.allocate_buffers(
            shape_dict={
                "images": frames.shape,
                "features": (frames.shape[0], 768),
            },
            device=self.device,
        )

        features = self.engine.infer(
            {"images": frames},
            torch.cuda.current_stream(self.device),
            use_cuda_graph=False,
        )["features"]

        # l2 normalize
        return F.normalize(features, p=2, dim=-1)
