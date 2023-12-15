import cvcuda
import torch

cvcuda_hsv_tensor = None


def rgb_to_hsv_nhwc_uint8(rgb_tensor: torch.Tensor) -> torch.Tensor:
    cvcuda_rgb_tensor = cvcuda.as_tensor(rgb_tensor, cvcuda.TensorLayout.NHWC)

    global cvcuda_hsv_tensor

    if cvcuda_hsv_tensor is None or cvcuda_rgb_tensor.shape != cvcuda_hsv_tensor.shape:
        cvcuda_hsv_tensor = cvcuda.Tensor(
            (rgb_tensor.shape[0], rgb_tensor.shape[1], rgb_tensor.shape[2], 3),
            cvcuda.Type.U8,
            cvcuda.TensorLayout.NHWC,
        )

    cvcuda.cvtcolor_into(
        cvcuda_hsv_tensor, cvcuda_rgb_tensor, cvcuda.ColorConversion.RGB2HSV
    )

    return torch.as_tensor(cvcuda_hsv_tensor.cuda(), device=rgb_tensor.device)
