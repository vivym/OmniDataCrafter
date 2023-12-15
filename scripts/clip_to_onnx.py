import clip
import onnx
import torch
import torch.nn as nn
from onnxconverter_common import float16
from onnxconverter_common import auto_mixed_precision as amp


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        clip_model, _ = clip.load("ViT-L/14", device=torch.device("cpu"), jit=False)
        self.visual = clip_model.visual

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images)


def main():
    use_amp = True
    device = torch.device("cuda")
    model = Model().to(device)

    images = torch.randn(1, 3, 224, 224, device=device)

    model_path = "./weights/clip_vit_l14.onnx"
    torch.onnx.export(
        model,
        (images,),
        model_path,
        input_names=["images"],
        output_names=["features"],
        dynamic_axes={
            "images": {0: "batch"},
            "features": {0: "batch"},
        },
    )

    model = onnx.load(model_path)
    if use_amp:
        model_fp16 = amp.auto_convert_mixed_precision(
            model,
            feed_dict={"images": images.cpu().numpy()},
            rtol=1e-2,
            atol=1e-2,
            keep_io_types=True,
        )
    else:
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    onnx.save(model_fp16, model_path.replace(".onnx", "_fp16.onnx"))


if __name__ == "__main__":
    main()
