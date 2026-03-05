import argparse
import csv
import os
from typing import Dict, List, Tuple, Optional


def _to_float(x: str) -> Optional[float]:
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


def _escape_latex_minimal(s: str) -> str:
    # minimal escaping for derived labels (paths) if user didn't provide LaTeX labels
    # NOTE: we intentionally do NOT escape backslashes so that labels may contain LaTeX like \xi.
    return (s
             .replace("_", "\\_")
             .replace("%", "\\%")
             .replace("&", "\\&")
             .replace("#", "\\#")
            )



import re


def _latexize_params_in_label(label: str) -> str:
    """Wrap parameter choices like 'h = ... , \\xi = ...' in $...$ for LaTeX tables."""
    s = label

    mh = re.search(r"\bh\s*=\s*([0-9.+\-eE]+)", s)
    mxi = re.search(r"\\xi\s*=\s*([0-9.+\-eE]+)", s)

    def norm(inner: str) -> str:
        inner = re.sub(r"\s*=\s*", "=", inner)
        inner = re.sub(r"\s*,\s*", ", ", inner)
        return inner

    # If we have both (in order), wrap the combined segment.
    if mh and mxi and mh.start() < mxi.start():
        seg = s[mh.start():mxi.end()]
        if "$" not in seg:
            return s[:mh.start()] + "$" + norm(seg) + "$" + s[mxi.end():]
        return s

    # Otherwise, wrap individually.
    if mh:
        seg = s[mh.start():mh.end()]
        if "$" not in seg:
            s = s[:mh.start()] + "$" + norm(seg) + "$" + s[mh.end():]

    mxi2 = re.search(r"\\xi\s*=\s*([0-9.+\-eE]+)", s)
    if mxi2:
        seg = s[mxi2.start():mxi2.end()]
        if "$" not in seg:
            s = s[:mxi2.start()] + "$" + norm(seg) + "$" + s[mxi2.end():]

    return s


def extract_eta_series(rows: List[Dict[str, str]]) -> Tuple[List[float], List[float], List[float]]:
    """Return (steps, c_log_vals, c_lin_vals) from rows that have non-empty eta coef columns."""
    steps, c_log_vals, c_lin_vals = [], [], []
    for row in rows:
        cl = _to_float(row.get("c_log_mean", ""))
        cm = _to_float(row.get("c_lin_mean", ""))
        if cl is not None or cm is not None:
            steps.append(float(int(row.get("step", "0"))))
            c_log_vals.append(cl if cl is not None else float('nan'))
            c_lin_vals.append(cm if cm is not None else float('nan'))
    return steps, c_log_vals, c_lin_vals


def summarize_val(rows: List[Dict[str, str]]) -> Tuple[Optional[float], Optional[float], float, int]:
    """
    Returns (last_val, best_val, wall_seconds, last_step).
    wall_seconds uses wall_cum_s if available, else integrates wall_dt_s across rows.
    """
    last_val: Optional[float] = None
    best_val: Optional[float] = None
    last_step = 0

    cum_wall = 0.0
    for row in rows:
        step = int(row.get("step", "0"))
        last_step = max(last_step, step)

        wall_cum = _to_float(row.get("wall_cum_s", ""))
        if wall_cum is not None:
            cum_wall = wall_cum
        else:
            wall_dt = _to_float(row.get("wall_dt_s", ""))
            if wall_dt is not None:
                cum_wall += wall_dt

        v = _to_float(row.get("val_loss", ""))
        if v is not None:
            last_val = v
            best_val = v if best_val is None else min(best_val, v)

    return last_val, best_val, float(cum_wall), int(last_step)


def print_latex_table(labels: List[str],
                      summaries: List[Tuple[Optional[float], Optional[float], float, int]],
                      caption: str,
                      labels_are_latex: bool) -> None:
    """
    Prints a LaTeX table to stdout.

    If labels_are_latex=True, labels are treated as raw LaTeX (no escaping). This is the
    recommended mode when you pass labels like 'Strang $h=0.01$' etc.
    """
    # Determine minima for bolding
    last_vals = [x[0] for x in summaries if x[0] is not None]
    best_vals = [x[1] for x in summaries if x[1] is not None]
    times = [x[2] for x in summaries]

    min_last = min(last_vals) if last_vals else None
    min_best = min(best_vals) if best_vals else None
    min_time = min(times) if times else None

    def fmt(v: Optional[float], m: Optional[float]) -> str:
        if v is None:
            return "-"
        s = f"{v:.4f}"
        return f"\\textbf{{{s}}}" if (m is not None and abs(v - m) <= 5e-7) else s

    def fmt_time(v: float) -> str:
        s = f"{v:.1f}"
        return f"\\textbf{{{s}}}" if (min_time is not None and abs(v - min_time) <= 5e-7) else s

    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \begin{tabular}{c|c|c|c}")
    print(r"        \textbf{model}       & \textbf{last} ($\downarrow$)      & \textbf{best} ($\downarrow$)     & \textbf{wall clock time} (seconds) ($\downarrow$) \\ \hline")
    for lab, (last_v, best_v, wall_s, _) in zip(labels, summaries):
        row_lab = lab if labels_are_latex else _escape_latex_minimal(_latexize_params_in_label(lab))
        print(f"        {row_lab}    & {fmt(last_v, min_last)}                   & {fmt(best_v, min_best)}                & {fmt_time(wall_s)} \\\\")
    print(r"    \end{tabular}")
    # caption is treated as raw LaTeX (so you can include math, macros, etc.)
    print(f"    \\caption{{{caption}}}")
    print(r"\end{table}")


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
    ap.add_argument("--latex_table", action="store_true", help="print a LaTeX summary table to stdout")
    ap.add_argument("--latex_caption", type=str, default=None, help="caption for LaTeX table (raw LaTeX allowed)")
    args = ap.parse_args()

    labels_are_latex = args.labels is not None

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

    labels: List[str] = []
    summaries: List[Tuple[Optional[float], Optional[float], float, int]] = []
    max_step_all = 0

    for i, run_dir in enumerate(args.runs):
        mpath = os.path.join(run_dir, "metrics.csv")
        if not os.path.exists(mpath):
            raise SystemExit(f"missing {mpath}")

        rows = read_metrics(mpath)
        xtr, ytr, xva, yva = extract_series(rows, args.xaxis)

        label = args.labels[i] if args.labels is not None else os.path.basename(run_dir.rstrip("/"))
        labels.append(label)

        last_val, best_val, wall_s, last_step = summarize_val(rows)
        summaries.append((last_val, best_val, wall_s, last_step))
        max_step_all = max(max_step_all, last_step)

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

    if args.latex_table:
        caption = args.latex_caption
        if caption is None:
            caption = f"Validation loss summary after {max_step_all} optimization steps."
        print_latex_table(labels, summaries, caption, labels_are_latex)

    # --- Eta coefficient plots (c_log, c_lin) ---
    # Collect eta data across all runs; skip if none have it.
    eta_data = []
    for i, run_dir in enumerate(args.runs):
        mpath = os.path.join(run_dir, "metrics.csv")
        rows = read_metrics(mpath)
        steps, c_log_vals, c_lin_vals = extract_eta_series(rows)
        label = args.labels[i] if args.labels is not None else os.path.basename(run_dir.rstrip("/"))
        eta_data.append((label, steps, c_log_vals, c_lin_vals))

    has_eta = any(len(d[1]) > 0 for d in eta_data)
    if has_eta:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for label, steps, c_log_vals, c_lin_vals in eta_data:
            if not steps:
                continue
            import math as _math
            valid_log = [v for v in c_log_vals if not _math.isnan(v)]
            valid_lin = [v for v in c_lin_vals if not _math.isnan(v)]
            if valid_log:
                ax1.plot(steps, c_log_vals, label=label)
            if valid_lin:
                ax2.plot(steps, c_lin_vals, label=label)
        ax1.set_xlabel("optimizer step")
        ax1.set_ylabel("$c_{\\log}$ (mean over layers)")
        ax1.set_title("Eta log coefficient $c_{\\log}$")
        ax1.legend(fontsize=7, frameon=True)
        ax2.set_xlabel("optimizer step")
        ax2.set_ylabel("$c_{\\mathrm{lin}}$ (mean over layers)")
        ax2.set_title("Eta linear coefficient $c_{\\mathrm{lin}}$")
        ax2.legend(fontsize=7, frameon=True)
        plt.tight_layout()
        base, ext = os.path.splitext(args.out)
        eta_out = base + "_eta" + ext
        plt.savefig(eta_out, dpi=180)
        plt.close()
        print(f"saved {eta_out}")


if __name__ == "__main__":
    main()
