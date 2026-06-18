"""Figure 1 — Dataset overview (ESKAPE bioactivity characteristics)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *  # noqa: F403,E402


def main():
    df_clean = load_eskape_clean()

    def _abbrev(name):
        parts = str(name).split()
        return f"$\\it{{{parts[0][0]}. {parts[1]}}}$" if len(parts) >= 2 else name

    org_counts = (df_clean.groupby(["target_organism", "active"]).size()
                  .unstack(fill_value=0).rename(columns={0: "Inactive", 1: "Active"}))
    org_counts.index = [_abbrev(x) for x in org_counts.index]
    org_sorted = org_counts.sort_values("Active", ascending=True)
    n_org = len(org_sorted)

    fig = plt.figure(figsize=(7.0, 4.0))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.02, right=0.98)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))

    # Panel A: class balance per organism
    y_pos = np.arange(n_org)
    ax1.barh(y_pos, org_sorted["Inactive"], height=0.55, color=C["inactive"],
             label="Inactive", zorder=3)
    ax1.barh(y_pos, org_sorted["Active"], height=0.55, left=org_sorted["Inactive"],
             color=C["active"], label="Active", zorder=3)
    ax1.set_yticks(y_pos); ax1.set_yticklabels(org_sorted.index, fontsize=6.5)
    ax1.set_xlabel("Compound count")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=6,
               frameon=False, handlelength=1.2, columnspacing=1.2, handletextpad=0.4)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    for i, (_, row) in enumerate(org_sorted.iterrows()):
        total = row["Inactive"] + row["Active"]
        ax1.text(total + total * 0.02, i, f"{total:,}", va="center", fontsize=5,
                 color=C["neutral"])
    add_panel_label(ax1, "A", x=-0.22)

    # Panel B: pChEMBL distribution
    pchem = df_clean["pchembl_value"].dropna().astype(float)
    if len(pchem) > 100:
        ax2.hist(pchem, bins=40, color=C["active"], edgecolor="white",
                 linewidth=0.25, alpha=0.85, zorder=3)
        ax2.axvline(5, color=C["inactive"], ls="--", lw=0.8,
                    label="pChEMBL = 5\n(10 μM threshold)")
        ax2.set_xlabel("pChEMBL value"); ax2.legend(fontsize=5.5, handlelength=1.2)
        med = pchem.median()
        ax2.axvline(med, color=C["neutral"], ls=":", lw=0.7)
        ax2.autoscale(enable=True, axis="y")
        ax2.text(med + 0.1, ax2.get_ylim()[1] * 0.92, f"median={med:.1f}",
                 fontsize=5.5, color=C["neutral"])
    else:
        log_vals = np.log10(df_clean["value_uM"].clip(1e-3, 1e5))
        ax2.hist(log_vals, bins=40, color=C["active"], edgecolor="white",
                 linewidth=0.25, alpha=0.85)
        ax2.set_xlabel("log₁₀(activity value, μM)")
    ax2.set_ylabel("Count"); add_panel_label(ax2, "B")

    # Panel C: assay type
    stype = df_clean["standard_type"].value_counts().head(6)
    bars = ax3.bar(np.arange(len(stype)), stype.values, color=C["neutral"],
                   edgecolor="none", width=0.6, zorder=3)
    ax3.set_xticks(np.arange(len(stype)))
    ax3.set_xticklabels(stype.index, rotation=35, ha="right", fontsize=6.5)
    ax3.set_ylabel("Count")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    for bar, val in zip(bars, stype.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + stype.max() * 0.02,
                 f"{val:,}", ha="center", va="bottom", fontsize=5)
    add_panel_label(ax3, "C")

    save_fig(fig, "figure_1_dataset_overview")


if __name__ == "__main__":
    main()
