import math

import torch
import PyNvCodec as nvc
import PytorchNvCodec as pnvc


class ColorSpaceConverter:
    """
    Colorspace conversion chain.
    """

    def __init__(self, width: int, height: int, gpu_id: int = 0):
        self.width = width
        self.height = height
        self.gpu_id = gpu_id
        self.chain: list[nvc.PySurfaceConverter] = []

    def add(self, src_fmt: nvc.PixelFormat, dst_fmt: nvc.PixelFormat) -> "ColorSpaceConverter":
        self.chain.append(
            nvc.PySurfaceConverter(self.width, self.height, src_fmt, dst_fmt, self.gpu_id)
        )
        return self

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        surf = src_surface
        cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

        for cvt in self.chain:
            surf: nvc.Surface = cvt.Execute(surf, cc)
            if surf.Empty():
                raise RuntimeError("Failed to perform color conversion")

        return surf.Clone(self.gpu_id)


def surface_to_tensor(surface: nvc.Surface) -> torch.Tensor:
    """
    Converts planar rgb surface to cuda float tensor.
    """

    if surface.Format() != nvc.PixelFormat.RGB_PLANAR:
        raise RuntimeError("Surface shall be of RGB_PLANAR pixel format")

    surf_plane = surface.PlanePtr()
    img_tensor: torch.Tensor = pnvc.DptrToTensor(
        surf_plane.GpuMem(),
        surf_plane.Width(),
        surf_plane.Height(),
        surf_plane.Pitch(),
        surf_plane.ElemSize(),
    )
    if img_tensor is None:
        raise RuntimeError("Can not export to tensor.")

    img_tensor.resize_(3, int(surf_plane.Height() / 3), surf_plane.Width())
    img_tensor = img_tensor.permute(1, 2, 0)
    # img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor)
    # img_tensor = torch.divide(img_tensor, 255.0)
    # img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

    return img_tensor


@torch.no_grad()
@torch.jit.script
def rgb_to_hsv(rgb_tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        rgb_tensor: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(rgb_tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rgb_tensor)}")

    if len(rgb_tensor.shape) < 3 or rgb_tensor.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {rgb_tensor.shape}")

    max_rgb, argmax_rgb = rgb_tensor.max(-3)
    min_rgb, argmin_rgb = rgb_tensor.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - rgb_tensor), dim=-3)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = 2.0 * math.pi * h  # we return 0 / 2pi output

    return torch.stack((h, s, v), dim=-3)
