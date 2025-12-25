import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


@dataclass
class Metrics:
    niqe: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    fid: Optional[float] = None
    lse_c: Optional[float] = None
    lse_d: Optional[float] = None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    max_frames: int = 300,
    stride: int = 1,
    resize: Optional[Tuple[int, int]] = (256, 256),
) -> List[Path]:
    """
    抽帧为 jpg：
    - max_frames: 最多抽取多少帧（防止评测时间爆炸）
    - stride: 每 stride 帧取 1 帧
    - resize: 统一分辨率，便于 SSIM/PSNR/FID（None 则不缩放）
    """
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    _ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法读取视频: {video_path}")

    frames: List[Path] = []
    idx = 0
    kept = 0
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            # OpenCV is BGR; save as jpg
            out = out_dir / f"{kept:06d}.jpg"
            cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frames.append(out)
            kept += 1
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"未抽取到任何帧: {video_path}")
    return frames


def _load_images_as_tensors(paths: List[Path], device: str = "cpu") -> torch.Tensor:
    # N,H,W,C (uint8) -> N,C,H,W float in [0,1]
    imgs = []
    for p in paths:
        arr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if arr is None:
            continue
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        imgs.append(arr)
    if not imgs:
        raise RuntimeError("读取图片失败：抽帧目录为空或损坏")
    x = np.stack(imgs, axis=0)
    t = torch.from_numpy(x).to(device=device, dtype=torch.float32) / 255.0
    t = t.permute(0, 3, 1, 2).contiguous()
    return t


def _compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    # pred/gt: N,C,H,W in [0,1]
    mse = torch.mean((pred - gt) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))
    return float(psnr.mean().item())


def _compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    # Prefer PIQ if available; otherwise fallback to skimage
    try:
        import piq

        val = piq.ssim(pred, gt, data_range=1.0, reduction="mean")
        return float(val.item())
    except Exception:
        from skimage.metrics import structural_similarity as ssim

        pred_np = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        gt_np = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        ssims = []
        for i in range(min(len(pred_np), len(gt_np))):
            ssims.append(
                ssim(gt_np[i], pred_np[i], channel_axis=2, data_range=255)
            )
        return float(np.mean(ssims))


def _compute_niqe(frames: torch.Tensor) -> float:
    """
    NIQE 通常在灰度图上计算。这里对每帧取灰度，然后取平均。
    """
    import piq

    # N,C,H,W -> N,1,H,W
    if frames.shape[1] == 3:
        r, g, b = frames[:, 0:1], frames[:, 1:2], frames[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = frames
    # PIQ niqe expects (N,1,H,W) in [0,1]
    scores = []
    for i in range(gray.shape[0]):
        scores.append(float(piq.niqe(gray[i : i + 1], data_range=1.0).item()))
    return float(np.mean(scores))


def _compute_fid_from_dirs(dir1: Path, dir2: Path, *, use_cuda: bool = False) -> float:
    """
    使用 torch-fidelity 计算 FID。
    说明：首次运行可能会下载 Inception 权重（取决于 torch-fidelity/torchvision 版本）。
    """
    from torch_fidelity import calculate_metrics

    metrics = calculate_metrics(
        input1=str(dir1),
        input2=str(dir2),
        fid=True,
        isc=False,
        kid=False,
        prc=False,
        cuda=use_cuda,
        verbose=False,
    )
    # key is "frechet_inception_distance"
    return float(metrics["frechet_inception_distance"])


def _maybe_compute_lse(*_: object, **__: object) -> Tuple[Optional[float], Optional[float]]:
    """
    LSE-C / LSE-D：
    - 需要口型同步网络（SyncNet/Wav2Lip eval）与音视频对齐策略。
    - 本仓库未内置官方权重与评测脚本（课程测试用例通常会提供）。
    这里返回 None，并在 README 里给出“如何接入/替换”的明确接口。
    """
    return None, None


def evaluate(gt_video: Path, pred_video: Path, *, stride: int, max_frames: int) -> Metrics:
    device = "cpu"

    with tempfile.TemporaryDirectory(prefix="eval_frames_") as tmp:
        tmp_dir = Path(tmp)
        gt_dir = tmp_dir / "gt"
        pred_dir = tmp_dir / "pred"

        gt_frames = _extract_frames(gt_video, gt_dir, stride=stride, max_frames=max_frames)
        pred_frames = _extract_frames(pred_video, pred_dir, stride=stride, max_frames=max_frames)

        # 对齐帧数（取最短）
        n = min(len(gt_frames), len(pred_frames))
        gt_frames = gt_frames[:n]
        pred_frames = pred_frames[:n]

        gt_t = _load_images_as_tensors(gt_frames, device=device)
        pred_t = _load_images_as_tensors(pred_frames, device=device)

        m = Metrics()
        m.psnr = _compute_psnr(pred_t, gt_t)
        m.ssim = _compute_ssim(pred_t, gt_t)
        m.niqe = _compute_niqe(pred_t)

        # FID：在抽帧目录上算（如需更严格，可改为使用整段视频的全部帧）
        try:
            m.fid = _compute_fid_from_dirs(pred_dir, gt_dir, use_cuda=False)
        except Exception as e:
            # 允许离线/缺权重场景继续跑其它指标
            m.fid = None
            print(f"[WARN] FID 计算失败（将返回 null）：{type(e).__name__}: {e}")

        m.lse_c, m.lse_d = _maybe_compute_lse(gt_video=gt_video, pred_video=pred_video)
        return m


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Talking-Face 视频评测（NIQE/PSNR/FID/SSIM/LSE-C/LSE-D）")
    p.add_argument("--gt_video", required=True, help="GT 视频路径（容器内路径）")
    p.add_argument("--pred_video", required=True, help="待评测视频路径（容器内路径）")
    p.add_argument("--out_json", default="", help="输出 json 路径（可选）")
    p.add_argument("--stride", type=int, default=2, help="抽帧步长（默认每 2 帧取 1 帧）")
    p.add_argument("--max_frames", type=int, default=300, help="最多抽取帧数（默认 300）")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    gt_video = Path(args.gt_video)
    pred_video = Path(args.pred_video)

    metrics = evaluate(gt_video, pred_video, stride=args.stride, max_frames=args.max_frames)
    payload: Dict[str, object] = asdict(metrics)

    # 控制台输出（docker 评测要求：直接输出指标）
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.out_json:
        out = Path(args.out_json)
        _ensure_dir(out.parent)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

