"""
Training Report — Llama 3.2 3B Phishing Detector

Reads trainer_state.json from the latest checkpoint and produces a
multi-panel PNG report saved alongside it.

Usage:
    D:\miniconda3\envs\mlenv\python.exe report.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# ----------------------------------------------------------------- paths
_HERE = Path(__file__).parent
_CKPT_DIR = _HERE.parent / "outputs" / "llama3.2-3b-phishing" / "checkpoints"
_OUT_DIR  = _HERE.parent / "outputs" / "llama3.2-3b-phishing"

# Use the latest checkpoint's trainer_state.json
_STATE_FILE = _CKPT_DIR / "checkpoint-1500" / "trainer_state.json"
_REPORT_PNG = _OUT_DIR / "training_report.png"

# ----------------------------------------------------------------- colours
C_TRAIN  = "#4C72B0"   # blue
C_EVAL   = "#DD8452"   # orange
C_LR     = "#55A868"   # green
C_GRAD   = "#C44E52"   # red
C_BEST   = "#8172B2"   # purple
C_BG     = "#F8F9FA"
C_GRID   = "#DEE2E6"
C_TEXT   = "#212529"


def load_history(path: Path):
    with open(path, encoding="utf-8") as f:
        state = json.load(f)

    train_steps, train_loss = [], []
    eval_steps,  eval_loss  = [], []
    lr_steps,    lr_vals    = [], []
    gn_steps,    gn_vals    = [], []

    for entry in state["log_history"]:
        step = entry["step"]
        if "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
            lr_steps.append(step)
            lr_vals.append(entry.get("learning_rate", 0))
            gn_steps.append(step)
            gn_vals.append(entry.get("grad_norm", 0))
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])

    best_step   = state["best_model_checkpoint"].split("checkpoint-")[-1]
    best_metric = state["best_metric"]
    meta = {
        "total_steps":   state["global_step"],
        "num_epochs":    state["num_train_epochs"],
        "best_step":     int(best_step),
        "best_eval_loss": best_metric,
    }
    return (
        (train_steps, train_loss),
        (eval_steps,  eval_loss),
        (lr_steps,    lr_vals),
        (gn_steps,    gn_vals),
        meta,
    )


# ----------------------------------------------------------------- eval metrics (from terminal output)
EVAL_METRICS = {
    "classes":   ["legitimate", "phishing"],
    "precision": [1.00,         1.00],
    "recall":    [1.00,         1.00],
    "f1":        [1.00,         1.00],
    "support":   [188,          312],
    "accuracy":  1.0000,
    "json_parse_rate": 1.0000,
    "n_samples": 500,
}


def smooth(values, window=5):
    """Simple moving-average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def make_report():
    (train_steps, train_loss), \
    (eval_steps,  eval_loss),  \
    (lr_steps,    lr_vals),    \
    (gn_steps,    gn_vals),    \
    meta = load_history(_STATE_FILE)

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(24, 18), facecolor=C_BG)
    fig.suptitle(
        "Llama 3.2 3B — Email Phishing Detector · Fine-tuning Report",
        fontsize=22, fontweight="bold", color=C_TEXT, y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.42, wspace=0.32,
        top=0.93, bottom=0.06, left=0.07, right=0.97,
    )

    # ── helpers ──────────────────────────────────────────────────────────
    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor(C_BG)
        ax.set_title(title, fontsize=14, fontweight="bold", color=C_TEXT, pad=10)
        ax.set_xlabel(xlabel, fontsize=11, color=C_TEXT)
        ax.set_ylabel(ylabel, fontsize=11, color=C_TEXT)
        ax.tick_params(colors=C_TEXT, labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(C_GRID)
        ax.grid(color=C_GRID, linestyle="--", linewidth=0.7, alpha=0.8)

    # ── Panel 1 (row 0, col 0–1 span): Training loss full run ────────────
    ax1 = fig.add_subplot(gs[0, :2])
    _style(ax1, "Training Loss", "Step", "Loss")

    train_arr = np.array(train_loss)
    smoothed   = smooth(train_arr, window=7)

    ax1.plot(train_steps, train_arr, color=C_TRAIN, alpha=0.25, linewidth=1.0, label="_raw")
    ax1.plot(train_steps, smoothed,  color=C_TRAIN, linewidth=2.5, label="Train loss (smoothed)")

    # overlay eval checkpoints
    ax1.scatter(eval_steps, eval_loss, color=C_EVAL, zorder=5, s=90, label="Val loss (checkpoints)")
    ax1.plot(eval_steps, eval_loss, color=C_EVAL, linewidth=2.0, linestyle="--", alpha=0.7)

    # mark best checkpoint
    best_idx = eval_steps.index(meta["best_step"])
    ax1.axvline(meta["best_step"], color=C_BEST, linestyle=":", linewidth=1.5, alpha=0.8)
    ax1.annotate(
        f"Best ckpt\nstep {meta['best_step']}\nloss {meta['best_eval_loss']:.4f}",
        xy=(meta["best_step"], eval_loss[best_idx]),
        xytext=(meta["best_step"] - 160, eval_loss[best_idx] + 0.004),
        fontsize=10, color=C_BEST,
        arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.5),
    )

    # epoch boundary lines
    steps_per_epoch = meta["total_steps"] // meta["num_epochs"]
    for e in range(1, meta["num_epochs"]):
        ax1.axvline(e * steps_per_epoch, color=C_GRID, linestyle="-", linewidth=1.2, alpha=0.9)
        ax1.text(e * steps_per_epoch + 8, max(train_arr) * 0.95,
                 f"Epoch {e}", fontsize=10, color="#6C757D")

    ax1.legend(fontsize=11, loc="upper right")
    ax1.set_xlim(0, meta["total_steps"] + 20)

    # ── Panel 2 (row 0, col 2): Eval loss zoom ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _style(ax2, "Validation Loss (per checkpoint)", "Step", "Eval Loss")

    ax2.plot(eval_steps, eval_loss, color=C_EVAL, linewidth=2, marker="o",
             markersize=6, markerfacecolor="white", markeredgewidth=1.8)
    ax2.scatter([meta["best_step"]], [meta["best_eval_loss"]],
                color=C_BEST, zorder=6, s=120, label=f"Best: {meta['best_eval_loss']:.4f}")
    ax2.legend(fontsize=10, loc="upper right")

    # annotate each point
    for s, v in zip(eval_steps, eval_loss):
        ax2.annotate(f"{v:.4f}", (s, v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color=C_TEXT)

    ax2.set_ylim(min(eval_loss) * 0.995, max(eval_loss) * 1.005)

    # ── Panel 3 (row 1, col 0–1): Learning rate schedule ─────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    _style(ax3, "Learning Rate Schedule (cosine decay with warmup)", "Step", "Learning Rate")

    ax3.plot(lr_steps, lr_vals, color=C_LR, linewidth=2)
    ax3.fill_between(lr_steps, lr_vals, alpha=0.12, color=C_LR)

    # mark warmup end
    warmup_end = next((s for s, v in zip(lr_steps, lr_vals) if v >= max(lr_vals) * 0.999), lr_steps[0])
    ax3.axvline(warmup_end, color=C_LR, linestyle=":", linewidth=1.5, alpha=0.7)
    ax3.text(warmup_end + 10, max(lr_vals) * 0.95, "Warmup end", fontsize=10, color=C_LR)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))

    # ── Panel 4 (row 1, col 2): Gradient norm ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    _style(ax4, "Gradient Norm", "Step", "Grad Norm")

    gn_arr   = np.array(gn_vals)
    gn_sm    = smooth(gn_arr, window=10)
    ax4.plot(gn_steps, gn_arr, color=C_GRAD, alpha=0.2, linewidth=0.9)
    ax4.plot(gn_steps, gn_sm,  color=C_GRAD, linewidth=2.2, label="Smoothed")
    ax4.legend(fontsize=10)

    # ── Panel 5 (row 2, col 0–1): Class metrics bar chart ────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_facecolor(C_BG)
    ax5.set_title("Test Set — Per-class Metrics  (500 samples)", fontsize=14,
                  fontweight="bold", color=C_TEXT, pad=10)
    ax5.spines[["top", "right"]].set_visible(False)
    ax5.spines[["left", "bottom"]].set_color(C_GRID)
    ax5.grid(axis="y", color=C_GRID, linestyle="--", linewidth=0.6, alpha=0.8)

    classes = EVAL_METRICS["classes"]
    metrics = ["precision", "recall", "f1"]
    colors  = [C_TRAIN, C_EVAL, C_LR]
    x       = np.arange(len(classes))
    width   = 0.22

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = EVAL_METRICS[metric]
        bars = ax5.bar(x + (i - 1) * width, vals, width,
                       label=metric.capitalize(), color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=C_TEXT)

    ax5.set_xticks(x)
    ax5.set_xticklabels(
        [f"{c}\n(n={EVAL_METRICS['support'][i]})" for i, c in enumerate(classes)],
        fontsize=12, color=C_TEXT,
    )
    ax5.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax5.set_ylim(0, 1.12)
    ax5.tick_params(colors=C_TEXT)
    ax5.legend(fontsize=11, loc="lower right")

    # ── Panel 6 (row 2, col 2): Summary stats box ────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(C_BG)
    ax6.axis("off")

    summary_lines = [
        ("MODEL",            "Llama 3.2 3B Instruct"),
        ("METHOD",           "QLoRA  (r=16, α=32, NF4)"),
        ("TRAIN SAMPLES",    "8,000"),
        ("EPOCHS",           f"{meta['num_epochs']}"),
        ("TOTAL STEPS",      f"{meta['total_steps']:,}"),
        ("BEST CHECKPOINT",  f"Step {meta['best_step']}"),
        ("BEST VAL LOSS",    f"{meta['best_eval_loss']:.4f}"),
        ("─" * 26,           ""),
        ("TEST ACCURACY",    "100.00 %"),
        ("JSON PARSE RATE",  "100.00 %"),
        ("TEST SAMPLES",     f"{EVAL_METRICS['n_samples']}"),
    ]

    y_pos = 0.97
    for label, value in summary_lines:
        if label.startswith("─"):
            ax6.axhline(y_pos + 0.01, xmin=0.0, xmax=1.0, color=C_GRID, linewidth=1)
            y_pos -= 0.04
            continue
        ax6.text(0.0,  y_pos, label, transform=ax6.transAxes,
                 fontsize=11, color="#6C757D", fontweight="bold", va="top")
        ax6.text(0.52, y_pos, value, transform=ax6.transAxes,
                 fontsize=11, color=C_TEXT, va="top")
        y_pos -= 0.085

    ax6.set_title("Run Summary", fontsize=14, fontweight="bold", color=C_TEXT, pad=10)
    ax6.patch.set_linewidth(1)
    ax6.patch.set_edgecolor(C_GRID)

    # ---------------------------------------------------------------- save
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_REPORT_PNG, dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Report saved → {_REPORT_PNG}")


if __name__ == "__main__":
    make_report()