"""Supplementary Figure S3 — Organism-specific RF performance."""
import sys
from pathlib import Path
from amr_repurposing.viz.common import *  # noqa: F403


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

    # Panel B: training-set size vs AUC, coloured by active fraction.
    # (AUC is already the y-axis, so colour encodes the *new* dimension — class balance.)
    active_frac = df_tr["n_active"] / df_tr["n_total"] * 100
    sc = axes[1].scatter(df_tr["n_total"], df_tr["roc_auc"], s=70, c=active_frac,
                         cmap="viridis", vmin=0, vmax=75, edgecolors="white",
                         linewidths=0.5, zorder=3)
    cb = fig.colorbar(sc, ax=axes[1], shrink=0.85, pad=0.03)
    cb.set_label("Active fraction (%)", fontsize=6); cb.ax.tick_params(labelsize=5.5)

    # Per-organism label offsets (points), hand-tuned to avoid overlap in the
    # 0.94 cluster; keyed by species epithet so order/sorting is irrelevant.
    OFF = {
        "baumannii": (-9, 7, "right"), "coli": (1, 10, "center"),
        "pneumoniae": (0, -12, "center"), "aeruginosa": (-9, -9, "right"),
        "cloacae": (0, 11, "center"), "aureus": (9, 0, "left"),
        "tuberculosis": (-9, 0, "right"), "faecium": (9, -2, "left"),
    }
    for _, row in df_tr.iterrows():
        dx, dy, ha = OFF.get(row["organism"].split()[1], (8, 0, "left"))
        axes[1].annotate(row["short"], xy=(row["n_total"], row["roc_auc"]),
                         xytext=(dx, dy), textcoords="offset points", fontsize=6,
                         ha=ha, va="center",
                         arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=0.4,
                                         shrinkA=0, shrinkB=3))
    axes[1].set_xlabel("Training samples"); axes[1].set_ylabel("ROC-AUC")
    axes[1].set_xlim(2200, 13000); axes[1].set_ylim(0.875, 0.99)
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    add_panel_label(axes[1], "B", x=-0.22)

    save_fig(fig, "figure_S3_organism_performance")


if __name__ == "__main__":
    main()
