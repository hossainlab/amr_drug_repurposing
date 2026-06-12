"""Figure 3 — Drug repurposing candidates (ranked + MLP/RF agreement)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *  # noqa: F403,E402

N_CAND = 20


def main():
    _, df_druglike = load_clean_candidates()
    top_n = df_druglike.head(N_CAND).copy()
    top_n["label"] = top_n["drug_name"].str.title().str.slice(0, 22)

    bar_colors = ["#2166AC" if s >= 0.8 else "#D6604D" if s >= 0.6 else "#969696"
                  for s in top_n["prob_ensemble"]]

    panel_h = max(0.32 * N_CAND + 0.8, 5.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, panel_h),
                             gridspec_kw={"wspace": 0.40, "width_ratios": [1.6, 1]})

    # Panel A: ranked horizontal bar
    y_pos = np.arange(len(top_n))
    axes[0].barh(y_pos, top_n["prob_ensemble"], height=0.65, color=bar_colors,
                 edgecolor="none", zorder=3)
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels(top_n["label"], fontsize=6.5)
    axes[0].invert_yaxis()
    axes[0].axvline(0.5, color=C["inactive"], ls="--", lw=0.7, zorder=4)
    axes[0].axvline(0.8, color=C["mlp"], ls="--", lw=0.7, zorder=4)
    axes[0].set_xlabel("Ensemble probability (MLP + RF)")
    axes[0].set_xlim(0, 1.06)
    axes[0].xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    for i, (_, row) in enumerate(top_n.iterrows()):
        p = row["prob_ensemble"]
        axes[0].text(p + 0.01, i, f"{p:.3f}", va="center", fontsize=5.5,
                     color="white" if p > 0.95 else "black")
    axes[0].legend(handles=[
        mpatches.Patch(color="#2166AC", label="p ≥ 0.8 (high confidence)"),
        mpatches.Patch(color="#D6604D", label="0.6 ≤ p < 0.8"),
        mpatches.Patch(color="#969696", label="p < 0.6"),
        plt.Line2D([0], [0], color=C["inactive"], ls="--", lw=0.9, label="p = 0.5 threshold"),
        plt.Line2D([0], [0], color=C["mlp"], ls="--", lw=0.9, label="p = 0.8 threshold"),
    ], fontsize=5.5, loc="lower right", handlelength=1.2)
    add_panel_label(axes[0], "A", x=-0.16)

    # Panel B: MLP vs RF agreement scatter
    axes[1].scatter(df_druglike["prob_mlp"], df_druglike["prob_rf"], s=3, alpha=0.2,
                    color=C["neutral"], linewidths=0, zorder=2)
    axes[1].scatter(top_n["prob_mlp"], top_n["prob_rf"], s=16, color=C["ensemble"],
                    zorder=4, linewidths=0, label=f"Top {N_CAND}")
    for _, row in top_n.head(5).iterrows():
        axes[1].annotate(row["label"][:16], xy=(row["prob_mlp"], row["prob_rf"]),
                         xytext=(5, 4), textcoords="offset points", fontsize=5.5,
                         arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=0.4))
    axes[1].plot([0, 1], [0, 1], color=C["neutral"], lw=0.6, ls=":", alpha=0.7, zorder=1)
    axes[1].set_xlabel("MLP probability"); axes[1].set_ylabel("RF probability")
    axes[1].set_xlim(-0.03, 1.03); axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(loc="upper left", fontsize=6)
    add_panel_label(axes[1], "B", x=-0.22)

    save_fig(fig, "figure_3_repurposing_candidates")


if __name__ == "__main__":
    main()
