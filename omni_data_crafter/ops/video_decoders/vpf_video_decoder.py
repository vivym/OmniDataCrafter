from typing import Generator

import cvcuda
import torch
import nvcv
import PyNvCodec as nvc
import PytorchNvCodec as pnvc

from .video_decoder import VideoDecoder


class VPFVideoDecoder(VideoDecoder):
    def __init__(
        self,
        video_path: str,
        batch_size: int,
        device_id: int = 0,
    ):
        super().__init__()

        self.video_path = video_path
        self.batch_size = batch_size
        self.device_id = device_id

        self.nv_demux = nvc.PyFFmpegDemuxer(video_path)
        self.fps = self.nv_demux.Framerate()
        self.total_frames = self.nv_demux.Numframes()
        self.width = self.nv_demux.Width()
        self.height = self.nv_demux.Height()

        self.pix_fmt = self.nv_demux.Format()
        is_yuv420 = (
            nvc.PixelFormat.YUV420 == self.pix_fmt
            or nvc.PixelFormat.NV12 == self.pix_fmt
        )
        is_yuv444 = nvc.PixelFormat.YUV444 == self.pix_fmt
        assert is_yuv420 or is_yuv444

        if is_yuv420:
            self.cvcuda_cvt_code = cvcuda.ColorConversion.YUV2RGB_NV12
        else:
            self.cvcuda_cvt_code = cvcuda.ColorConversion.YUV2RGB

        codec = self.nv_demux.Codec()
        is_hevc = nvc.CudaVideoCodec.HEVC == codec

        if is_hevc and is_yuv444:
            raise NotImplementedError(
                "YUV444 HW decode is supported for HEVC only."
            )

        self.cvcuda_rgb_tensor = None
        self.cvcuda_hsv_tensor = None

    def _batch_cvt(self, yuv_tensors: list[torch.Tensor], pixel_format: str) -> torch.Tensor:
        # b, h, w, c
        yuv_tensor = torch.stack(yuv_tensors, dim=0)[..., None]
        cvcuda_yuv_tensor = cvcuda.as_tensor(yuv_tensor, cvcuda.TensorLayout.NHWC)

        if self.cvcuda_rgb_tensor is None or yuv_tensor.shape[0] != self.cvcuda_rgb_tensor.shape[0]:
            self.cvcuda_rgb_tensor = cvcuda.Tensor(
                (yuv_tensor.shape[0], self.height, self.width, 3),
                nvcv.Type.U8,
                nvcv.TensorLayout.NHWC,
            )

        cvcuda.cvtcolor_into(self.cvcuda_rgb_tensor, cvcuda_yuv_tensor, self.cvcuda_cvt_code)

        if pixel_format == "hsv":
            if self.cvcuda_hsv_tensor is None or yuv_tensor.shape[0] != self.cvcuda_hsv_tensor.shape[0]:
                self.cvcuda_hsv_tensor = cvcuda.Tensor(
                    (yuv_tensor.shape[0], self.height, self.width, 3),
                    nvcv.Type.U8,
                    nvcv.TensorLayout.NHWC,
                )

            cvcuda.cvtcolor_into(
                self.cvcuda_hsv_tensor, cvcuda_yuv_tensor, cvcuda.ColorConversion.RGB2HSV
            )
            return torch.as_tensor(self.cvcuda_hsv_tensor.cuda(), device=yuv_tensor.device)
        else:
            return torch.as_tensor(self.cvcuda_rgb_tensor.cuda(), device=yuv_tensor.device)

    def iter_frames(self, pixel_format: str = "rgb") -> Generator[torch.Tensor, None, None]:
        assert pixel_format in ("rgb", "hsv")

        nv_decoder = nvc.PyNvDecoder(self.video_path, self.device_id)

        yuv_tensors = []

        while True:
            # Decode NV12 surface
            surface: nvc.Surface = nv_decoder.DecodeSingleSurface()
            if surface.Empty():
                break

            surf_plane = surface.PlanePtr()
            yuv_tensor = pnvc.makefromDevicePtrUint8(
                surf_plane.GpuMem(),
                surf_plane.Width(),
                surf_plane.Height(),
                surf_plane.Pitch(),
                surf_plane.ElemSize(),
            )
            yuv_tensors.append(yuv_tensor)

            if len(yuv_tensors) >= self.batch_size:
                yield self._batch_cvt(yuv_tensors, pixel_format)
                yuv_tensors = []

        if len(yuv_tensors) > 0:
            yield self._batch_cvt(yuv_tensors, pixel_format)
