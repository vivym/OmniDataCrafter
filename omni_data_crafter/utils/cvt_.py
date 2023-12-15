import torch
import triton
import triton.language as tl

ITUR_BT_601_CY = 1220542
ITUR_BT_601_CVR = 1673527
ITUR_BT_601_CVG = -852492
ITUR_BT_601_CUG = -409993
ITUR_BT_601_CUB = 2116026
ITUR_BT_601_SHIFT = 20


@triton.jit
def yuv420sp_to_rgb_nhwc_kernel(
    src_ptr,
    dst_ptr,
    batch_size,
    src_height,
    src_width,
    dst_height,
    dst_width,
    ITUR_BT_601_CY: tl.constexpr,
    ITUR_BT_601_CVR: tl.constexpr,
    ITUR_BT_601_CVG: tl.constexpr,
    ITUR_BT_601_CUG: tl.constexpr,
    ITUR_BT_601_CUB: tl.constexpr,
    ITUR_BT_601_SHIFT: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0)
    dst_y = tl.program_id(axis=1)
    dst_x = tl.program_id(axis=2)

    if batch_idx >= batch_size:
        return

    if dst_y >= dst_height:
        return

    if dst_x >= dst_width:
        return

    uv_x = dst_x if dst_x % 2 == 0 else dst_x - 1

    y = tl.load(src_ptr + batch_idx * src_height * src_width + dst_y * src_width + dst_x)
    u = tl.load(src_ptr + batch_idx * src_height * src_width + (dst_height + dst_y // 2) * src_width + uv_x)
    v = tl.load(src_ptr + batch_idx * src_height * src_width + (dst_height + dst_y // 2) * src_width + uv_x + 1)

    y_int32 = y.to(tl.int32)
    u_int32 = u.to(tl.int32)
    v_int32 = v.to(tl.int32)

    yy = tl.where(y_int32 <= 16, 0, y_int32 - 16) * ITUR_BT_601_CY
    uu = u_int32 - 128
    vv = v_int32 - 128

    r = yy + ITUR_BT_601_CVR * vv
    g = yy + ITUR_BT_601_CVG * vv + ITUR_BT_601_CUG * uu
    b = yy + ITUR_BT_601_CUB * uu

    r = (r + (1 << (ITUR_BT_601_SHIFT - 1))) >> ITUR_BT_601_SHIFT
    g = (g + (1 << (ITUR_BT_601_SHIFT - 1))) >> ITUR_BT_601_SHIFT
    b = (b + (1 << (ITUR_BT_601_SHIFT - 1))) >> ITUR_BT_601_SHIFT

    dst_idx = batch_idx * dst_height * dst_width * 3 + dst_y * dst_width * 3 + dst_x * 3
    tl.store(dst_ptr + dst_idx, r)
    tl.store(dst_ptr + dst_idx + 1, g)
    tl.store(dst_ptr + dst_idx + 2, b)


def yuv420sp_to_rgb_nhwc(src: torch.Tensor):
    batch_size = src.shape[0]
    src_height = src.shape[1]
    src_width = src.shape[2]

    dst_height = src_height * 2 // 3
    dst_width = src_width

    dst = torch.empty(
        (batch_size, dst_height, dst_width, 3),
        dtype=src.dtype,
        device=src.device,
    )

    grid = lambda x: (x.shape[0], x.shape[1], x.shape[2])

    yuv420sp_to_rgb_nhwc_kernel[grid(src)](
        src.contiguous(),
        dst.contiguous(),
        batch_size,
        src_height,
        src_width,
        dst_height,
        dst_width,
        ITUR_BT_601_CY,
        ITUR_BT_601_CVR,
        ITUR_BT_601_CVG,
        ITUR_BT_601_CUG,
        ITUR_BT_601_CUB,
        ITUR_BT_601_SHIFT,
    )

    return dst


def main():
    src = torch.ones((1, 1080, 1920), dtype=torch.uint8, device="cuda")
    dst = yuv420sp_to_rgb_nhwc(src)
    print(dst.shape)
    print("dst", dst.float().mean())
    print(yuv420sp_to_rgb_nhwc_kernel.cache)


if __name__ == "__main__":
    main()
