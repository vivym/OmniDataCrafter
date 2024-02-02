from typing import Generator

import numpy as np
import torch
import PyNvCodec as nvc

from omni_data_crafter.utils.vpf import ColorSpaceConverter, surface_to_tensor, rgb_to_hsv


class VPFStream:
    def __init__(
        self,
        video_path: str,
        device_idx: int = 0,
    ):
        super().__init__()

        self.video_path = video_path
        self.device_idx = device_idx

        self.nv_demux = nvc.PyFFmpegDemuxer(video_path)
        self.fps = self.nv_demux.Framerate()
        self.total_frames = self.nv_demux.Numframes()
        self.width = self.nv_demux.Width()
        self.height = self.nv_demux.Height()

    def decode(self, pixel_format: str = "hsv") -> Generator[torch.Tensor, None, None]:
        assert pixel_format in ("rgb", "hsv")

        nv_decoder = nvc.PyNvDecoder(self.video_path, self.device_idx)

        # TODO: batch convert (CV-CUDA)
        to_rgb = ColorSpaceConverter(self.width, self.height, self.device_idx)
        to_rgb.add(nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420)
        to_rgb.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB)
        to_rgb.add(nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR)

        while True:
            # Decode NV12 surface
            surface: nvc.Surface = nv_decoder.DecodeSingleSurface()
            if surface.Empty():
                break

            # yuv420_tensor = surface_to_tensor(surface)
            surf_plane = surface.PlanePtr()
            import PytorchNvCodec as pnvc
            yuv420_tensor = pnvc.makefromDevicePtrUint8(
                surf_plane.GpuMem(),
                surf_plane.Width(),
                surf_plane.Height(),
                surf_plane.Pitch(),
                surf_plane.ElemSize(),
            )

            print("yuv420_tensor", yuv420_tensor.shape)

            from omni_data_crafter.utils.cvt import yuv420sp_to_rgb_nhwc
            rgb_tensor2 = yuv420sp_to_rgb_nhwc(yuv420_tensor[None])[0]

            print("rgb_tensor2", rgb_tensor2.shape)

            import cvcuda
            import nvcv
            yuv420_tensor = yuv420_tensor[None, ..., None]
            cvcuda_YUVtensor = cvcuda.as_tensor(yuv420_tensor, nvcv.TensorLayout.NHWC)
            cvcuda_code = cvcuda.ColorConversion.YUV2RGB_NV12
            cvcuda_RGBtensor_batch = cvcuda.Tensor(
                (1, rgb_tensor2.shape[0], rgb_tensor2.shape[1], 3),
                nvcv.Type.U8,
                nvcv.TensorLayout.NHWC,
            )

            cvcuda.cvtcolor_into(
                cvcuda_RGBtensor_batch, cvcuda_YUVtensor, cvcuda_code
            )

            rgb_tensor3 = torch.as_tensor(cvcuda_RGBtensor_batch.cuda())[0]
            print("rgb_tensor3", rgb_tensor3.shape, rgb_tensor3.dtype, rgb_tensor3.device)

            # Convert to planar RGB
            rgb_pln = to_rgb.run(surface)
            if rgb_pln.Empty():
                break

            rgb_tensor = surface_to_tensor(rgb_pln)
            rgb_tensor = rgb_tensor.permute(1, 2, 0)

            print("rgb_tensor", rgb_tensor.shape)

            diff = (rgb_tensor.float() - rgb_tensor2.float()).abs()
            diff2 = (rgb_tensor.float().cpu() - rgb_tensor3.float()).abs()

            rgb_array = rgb_tensor.cpu().numpy()
            from PIL import Image
            Image.fromarray(rgb_array).save("rgb.png")

            rgb_array = rgb_tensor2.cpu().numpy()
            from PIL import Image
            Image.fromarray(rgb_array).save("rgb2.png")

            rgb_array = rgb_tensor3.cpu().numpy()
            from PIL import Image
            Image.fromarray(rgb_array).save("rgb3.png")

            print("diff", diff.mean(), diff.max(), diff.min())
            print("count #1", (diff > 0.5).sum(), diff.numel())
            print("diff2", diff2.mean(), diff2.max(), diff2.min())
            exit(0)

            if pixel_format == "rgb":
                yield rgb_tensor
            elif pixel_format == "hsv":
                yield rgb_to_hsv(rgb_tensor)


class VPFDemuxStream(VPFStream):
    def iter_frames(self, pixel_format: str = "hsv") -> Generator[torch.Tensor, None, None]:
        assert pixel_format in ("rgb", "hsv")

        nv_decoder = nvc.PyNvDecoder(self.video_path, self.device_idx)

        # TODO: batch convert (CV-CUDA)
        to_rgb = ColorSpaceConverter(self.width, self.height, self.device_idx)
        to_rgb.add(nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420)
        to_rgb.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB)
        to_rgb.add(nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR)

        packet = np.ndarray(shape=(0), dtype=np.uint8)
        pdata_in, pdata_out = nvc.PacketData(), nvc.PacketData()

        while True:
            # Demuxer has sync design, it returns packet every time it's called.
            # If demuxer can't return packet it usually means EOF.
            if not self.nv_demux.DemuxSinglePacket(packet):
                break

            # Get last packet data to obtain frame timestamp
            self.nv_demux.LastPacketData(pdata_in)

            # Decoder is async by design.
            # As it consumes packets from demuxer one at a time it may not return
            # decoded surface every time the decoding function is called.
            surface: nvc.Surface = nv_decoder.DecodeSurfaceFromPacket(pdata_in, packet, pdata_out)
            if not surface.Empty():
                # Convert to planar RGB
                rgb_pln = to_rgb.run(surface)
                if rgb_pln.Empty():
                    break

                rgb_tensor = surface_to_tensor(rgb_pln)

                if pixel_format == "rgb":
                    yield rgb_tensor
                elif pixel_format == "hsv":
                    yield rgb_to_hsv(rgb_tensor)

        # Now we flush decoder to emtpy decoded frames queue.
        while True:
            surface: nvc.Surface = nv_decoder.FlushSingleSurface()
            if surface.Empty():
                break

            # Convert to planar RGB
            rgb_pln = to_rgb.run(surface)
            if rgb_pln.Empty():
                break

            rgb_tensor = surface_to_tensor(rgb_pln)

            if pixel_format == "rgb":
                yield rgb_tensor
            elif pixel_format == "hsv":
                yield rgb_to_hsv(rgb_tensor)
