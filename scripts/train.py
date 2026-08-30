#!/usr/bin/env python
"""Train one segmentation model and write predictions for the evaluator.

Every architecture gets the same budget — same epochs, batch size, optimiser,
schedule, augmentation and seed. That is the whole methodological point: the
source notebooks tuned each model separately and then compared the results,
which makes the comparison uninterpretable. Anything tuned per model here would
reintroduce that problem.

    python scripts/train.py --root /path/to/CamVid --model fcn8s --epochs 40

Writes:
  checkpoints/<model>/best.pt          weights at best validation mIoU
  predictions/<model>/<sample>.png     val-split index masks for evaluate.py
  reports/train/<model>.json           the run's config, curve and environment

Then:

    python scripts/evaluate.py --root ... --predictions fcn8s=predictions/fcn8s
    python scripts/report.py

Requires the `train` extra (torch, torchvision, pillow).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from drivemetrics import ConfusionMatrix, dataset_iou, safety_recall  # noqa: E402
from drivemetrics.models.registry import MODEL_INFO, available_models  # noqa: E402
from drivemetrics.taxonomy import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, required=True, help="CamVid root")
    p.add_argument("--model", required=True, choices=available_models())
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--class-weights",
        choices=["none", "inverse-sqrt"],
        default="none",
        help=(
            "'none' is plain cross-entropy — the condition under which the "
            "observed models went blind to pedestrians, and the default the "
            "comparison starts from. 'inverse-sqrt' is the obvious remedy and "
            "is offered so the effect can be measured rather than assumed."
        ),
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT)
    p.add_argument("--limit", type=int, default=None, help="truncate split (smoke tests)")
    return p.parse_args(argv)


def set_determinism(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN costs some speed and buys reproducible numbers. For a
    # repository whose subject is untrustworthy results, that is the right trade.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(root: Path, mode: str):
    """Inverse-square-root frequency weights, or None for plain cross-entropy."""
    if mode == "none":
        return None
    import torch

    from drivemetrics.data.camvid import class_pixel_distribution, load_split

    stats = class_pixel_distribution(load_split(root, "train"))
    shares = np.array(
        [stats["per_class"][n]["pixel_share"] or 0.0 for n in CLASS_NAMES], dtype=np.float64
    )
    shares = np.clip(shares, 1e-6, None)
    w = 1.0 / np.sqrt(shares)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from drivemetrics.data.dataset import CamVidDataset, decode_prediction
    from drivemetrics.models.registry import build_model

    if not args.root.exists():
        print(f"error: CamVid root not found: {args.root}", file=sys.stderr)
        return 2

    set_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    train_ds = CamVidDataset(args.root, "train", train=True, seed=args.seed)
    val_ds = CamVidDataset(args.root, "val", train=False, seed=args.seed)
    if args.limit:
        train_ds.split.samples = train_ds.split.samples[: args.limit]
        val_ds.split.samples = val_ds.split.samples[: args.limit]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
    )

    model = build_model(args.model, NUM_CLASSES, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    weights = compute_class_weights(args.root, args.class_weights)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        ignore_index=IGNORE_INDEX,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    print(f"{args.model}: {n_params / 1e6:.1f}M params on {device}")
    print(f"  {len(train_ds)} train / {len(val_ds)} val, {args.epochs} epochs, "
          f"batch {args.batch_size}, lr {args.lr}, class-weights={args.class_weights}")
    print(f"  {MODEL_INFO[args.model]['note']}")

    ckpt_dir = args.out / "checkpoints" / args.model
    pred_dir = args.out / "predictions" / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best = {"mean_iou": -1.0, "epoch": -1}
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total, seen = 0.0, 0
        for images, masks, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * images.size(0)
            seen += images.size(0)
        scheduler.step()
        train_loss = total / max(seen, 1)

        # Validate with the same ConfusionMatrix the reporting pipeline uses, so
        # the number printed here and the number in the final report are the
        # same quantity computed by the same code.
        model.eval()
        cm = ConfusionMatrix(NUM_CLASSES, IGNORE_INDEX)
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(images)
                cm.update(masks.numpy(), decode_prediction(logits.float()))

        iou = dataset_iou(cm)
        vru = safety_recall(cm)["pooled"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "mean_iou": None if np.isnan(iou.mean) else float(iou.mean),
                "pixel_accuracy": cm.pixel_accuracy(),
                "vru_recall": None if np.isnan(vru) else float(vru),
                "lr": scheduler.get_last_lr()[0],
            }
        )
        flag = ""
        if iou.mean > best["mean_iou"]:
            best = {"mean_iou": float(iou.mean), "epoch": epoch}
            torch.save(
                {"model": args.model, "state_dict": model.state_dict(),
                 "epoch": epoch, "mean_iou": float(iou.mean), "seed": args.seed},
                ckpt_dir / "best.pt",
            )
            flag = "  *"
        print(
            f"  epoch {epoch:3d}  loss {train_loss:.4f}  mIoU {iou.mean:.4f}"
            f"  pixAcc {cm.pixel_accuracy():.4f}  VRU recall {vru:.3f}{flag}"
        )

    # -- write val predictions from the best checkpoint --------------------
    from PIL import Image

    state = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        for images, _, ids in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(images)
            for pred, sid in zip(decode_prediction(logits.float()), ids):
                Image.fromarray(pred, mode="L").save(pred_dir / f"{sid}.png")

    elapsed = time.time() - start
    record = {
        "model": args.model,
        "model_info": MODEL_INFO[args.model],
        "n_parameters": int(n_params),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "seed": args.seed,
            "pretrained": not args.no_pretrained,
            "class_weights": args.class_weights,
            "amp": amp,
            "augmentation": "hflip p=0.5, brightness/contrast jitter 0.2",
        },
        "best": best,
        "history": history,
        "wall_seconds": elapsed,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "predictions_dir": str(pred_dir.relative_to(args.out)).replace("\\", "/"),
    }
    out_json = args.out / "reports" / "train" / f"{args.model}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"\nbest mIoU {best['mean_iou']:.4f} at epoch {best['epoch']} "
          f"({elapsed / 60:.1f} min)")
    print(f"wrote {len(val_ds)} predictions to {pred_dir}")
    print(f"wrote {out_json}")
    print(f"\nnext:\n  python scripts/evaluate.py --root {args.root} "
          f"--predictions {args.model}={pred_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
