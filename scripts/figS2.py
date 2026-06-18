"""Supplementary Figure S2 — ChemBERTa fine-tuning loss."""
import sys
import glob
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *  # noqa: F403,E402


def main():
    logs = sorted(glob.glob(str(CKPT_DIR / "chembert" / "checkpoint-*" / "trainer_state.json")))
    if not logs:
        print(f"  No ChemBERTa trainer_state.json in {CKPT_DIR / 'chembert'}")
        return
    # pick the checkpoint with the most logged history (latest full run)
    state = max((json.load(open(p)) for p in logs), key=lambda s: len(s.get("log_history", [])))
    history = state.get("log_history", [])
    train_loss = [(h["step"], h["loss"]) for h in history if "loss" in h and "eval_loss" not in h]
    eval_loss = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    if train_loss:
        s, v = zip(*train_loss)
        ax.plot(s, v, color=C["mlp"], lw=0.9, alpha=0.75, label="Training loss", zorder=3)
    if eval_loss:
        s, v = zip(*eval_loss)
        ax.plot(s, v, color=C["rf"], lw=1.4, ls="--", marker="o", ms=3.5,
                markeredgewidth=0, label="Validation loss", zorder=4)
        best_step = s[int(np.argmin(v))]
        ax.axvline(best_step, color=C["neutral"], lw=0.7, ls=":", alpha=0.8,
                   label=f"Best ckpt (step {best_step:,})", zorder=2)
    ax.set_xlabel("Training steps"); ax.set_ylabel("Cross-entropy loss")
    ax.legend(fontsize=5.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    save_fig(fig, "figure_S2_chemberta_loss")


if __name__ == "__main__":
    main()
