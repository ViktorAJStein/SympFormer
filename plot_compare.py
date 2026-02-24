import argparse
import csv
import os
from typing import Dict, List, Tuple


def _to_float(x: str):
    return float(x) if x is not None and x != "" else None


def read_metrics(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def extract_series(rows: List[Dict[str, str]], xaxis: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Returns (x_train, y_train, x_val, y_val).

    xaxis:
      - step: optimizer step index
      - wall: cumulative wall time in seconds (prefers wall_cum_s; else integrates wall_dt_s)
      - tokens: cumulative tokens processed (requires tokens_cum)
    """
    x_train: List[float] = []
    y_train: List[float] = []
    x_val: List[float] = []
    y_val: List[float] = []

    cum_wall = 0.0

    for row in rows:
        step = int(row.get("step", "0"))
        train_loss = _to_float(row.get("train_loss", ""))
        val_loss = _to_float(row.get("val_loss", ""))

        # Determine x for this row
        if xaxis == "step":
            x = float(step)
        elif xaxis == "wall":
            wall_cum = _to_float(row.get("wall_cum_s", ""))
            if wall_cum is not None:
                cum_wall = wall_cum
            else:
                wall_dt = _to_float(row.get("wall_dt_s", ""))
                if wall_dt is not None:
                    cum_wall += wall_dt
            x = float(cum_wall)
        elif xaxis == "tokens":
            tok = row.get("tokens_cum", "")
            if tok == "":
                raise ValueError("tokens_cum missing in metrics.csv; re-run training with updated train.py")
            x = float(int(tok))
        else:
            raise ValueError(f"unknown xaxis={xaxis}")

        if train_loss is not None:
            x_train.append(x)
            y_train.append(train_loss)
        if val_loss is not None:
            x_val.append(x)
            y_val.append(val_loss)

    return x_train, y_train, x_val, y_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories (each must contain metrics.csv)")
    ap.add_argument("--labels", nargs="+", default=None, help="Optional labels (same length as --runs)")
    ap.add_argument("--xaxis", type=str, default="step", choices=["step", "wall", "tokens"],
                    help="x-axis: optimizer steps, cumulative wall time, or cumulative tokens")
    ap.add_argument("--out", type=str, default="compare_loss.png")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--val_only", action="store_true", help="plot only validation points")
    ap.add_argument("--annotate_last", action="store_true", help="write labels near last points")
    args = ap.parse_args()

    if args.labels is not None and len(args.labels) != len(args.runs):
        raise SystemExit("--labels must have the same length as --runs")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"matplotlib not available: {e}")

    if args.title is None:
        if args.xaxis == "step":
            title = "Loss vs optimizer step"
        elif args.xaxis == "wall":
            title = "Loss vs wall time (s)"
        else:
            title = "Loss vs tokens"
    else:
        title = args.title

    plt.figure(figsize=(8, 4.8))

    for i, run_dir in enumerate(args.runs):
        mpath = os.path.join(run_dir, "metrics.csv")
        if not os.path.exists(mpath):
            raise SystemExit(f"missing {mpath}")

        rows = read_metrics(mpath)
        xtr, ytr, xva, yva = extract_series(rows, args.xaxis)

        label = args.labels[i] if args.labels is not None else os.path.basename(run_dir.rstrip("/"))

        if (not args.val_only) and xtr:
            plt.plot(xtr, ytr, label=f"{label}:train")
            if args.annotate_last:
                plt.text(xtr[-1], ytr[-1], f" {label}:train", fontsize=8)

        if xva:
            plt.scatter(xva, yva, s=22, label=f"{label}:val")
            if args.annotate_last:
                plt.text(xva[-1], yva[-1], f" {label}:val", fontsize=8)

    if args.xaxis == "step":
        plt.xlabel("optimizer step")
    elif args.xaxis == "wall":
        plt.xlabel("cumulative wall time (s)")
    else:
        plt.xlabel("cumulative tokens")

    plt.ylabel("loss")
    plt.title(title)
    plt.legend(fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
