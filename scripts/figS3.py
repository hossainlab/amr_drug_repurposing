"""Supplementary Figure S3 — Organism-specific RF performance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *  # noqa: F403,E402


def main():
    path = DATA_DIR / "organism_rf_results.csv"
    if not path.exists():
        print(f"  {path} not found — run nb02 organism section.")
        return
    df_org = pd.read_csv(path)
    if df_org.empty or not df_org["status"].eq("trained").any():
        print("  No trained organism models.")
        return

    df_tr = df_org[df_org["status"] == "trained"].copy()
    df_tr["short"] = df_tr["organism"].apply(
        lambda x: f"$\\it{{{x.split()[0][0]}. {x.split()[1]}}}$" if len(x.split()) >= 2 else x)
    df_tr = df_tr.sort_values("roc_auc", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"wspace": 0.46})

    # Panel A: ROC-AUC bar per organism
    y_pos = np.arange(len(df_tr))
    hbars = axes[0].barh(y_pos, df_tr["roc_auc"], height=0.55, color=C["active"],
                         edgecolor="none", zorder=3)
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels(df_tr["short"], fontsize=7)
    axes[0].axvline(0.5, color=C["neutral"], ls=":", lw=0.7, zorder=2)
    axes[0].set_xlabel("ROC-AUC"); axes[0].set_xlim(0.45, 1.02)
    for bar, val in zip(hbars, df_tr["roc_auc"]):
        axes[0].text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                     va="center", fontsize=5.5)
    add_panel_label(axes[0], "A", x=-0.22)

    # Panel B: size vs AUC bubble
    active_frac = df_tr["n_active"] / df_tr["n_total"]
    sc = axes[1].scatter(df_tr["n_total"], df_tr["roc_auc"], s=active_frac * 180 + 20,
                         c=df_tr["roc_auc"], cmap="RdYlGn", vmin=0.5, vmax=1.0,
                         edgecolors="none", alpha=0.9, zorder=3)
    for _, row in df_tr.iterrows():
        axes[1].annotate(row["short"], xy=(row["n_total"], row["roc_auc"]),
                         xytext=(5, 3), textcoords="offset points", fontsize=5.5, ha="left")
    cb = fig.colorbar(sc, ax=axes[1], shrink=0.85, pad=0.03)
    cb.set_label("ROC-AUC", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    axes[1].axhline(0.5, color=C["neutral"], ls=":", lw=0.7, zorder=2)
    axes[1].set_xlabel("Training samples"); axes[1].set_ylabel("ROC-AUC")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    add_panel_label(axes[1], "B", x=-0.22)
    for frac, label in [(0.10, "10%"), (0.40, "40%"), (0.70, "70% active")]:
        axes[1].scatter([], [], s=frac * 180 + 20, color=C["neutral"], alpha=0.6,
                        label=label, edgecolors="none")
    axes[1].legend(title="Active fraction", fontsize=5.5, title_fontsize=5.5,
                   loc="lower right", handletextpad=0.4, labelspacing=0.5)

    save_fig(fig, "figure_S3_organism_performance")


if __name__ == "__main__":
    main()
