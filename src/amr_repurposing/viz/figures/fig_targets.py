"""Figure — Target-resolved repurposing.

Adds the mechanistic layer requested in team review: per-target QSAR performance,
a candidate x target activity heatmap, and the antifolate -> folate-pathway worked
example that explains the top pooled hits.
Panels:
  A  per-target cross-validated ROC-AUC (n molecules annotated; exploratory shaded)
  B  top candidates x target predicted-activity heatmap
  C  folate-pathway candidates (DHFR / thymidylate synthase) by target score
"""
import sys
from pathlib import Path
from amr_repurposing.viz.common import *  # noqa: F403


def _short(name: str, n: int = 26) -> str:
    name = (name.replace("Enoyl-[acyl-carrier-protein] reductase [NADH]", "Enoyl-ACP reductase (InhA)")
                .replace("Enoyl-[acyl-carrier-protein] reductase [NADPH] FabI", "Enoyl-ACP reductase (FabI)")
                .replace("Mycobacterial beta-ketoacyl-[acyl-carrier-protein] synthase III", "Beta-ketoacyl-ACP synthase (KasA)")
                .replace("Metallo-beta-lactamase type 2", "Metallo-beta-lactamase")
                .replace("Phosphotyrosine-protein phosphatase PTPB", "Protein phosphatase PtpB")
                .replace("Isoleucine--tRNA ligase", "Ile-tRNA ligase"))
    return name if len(name) <= n else name[:n - 1] + "…"


def main():
    metrics = load_target_metrics().sort_values("cv_roc_auc")
    screen = load_target_screen()
    cand, _ = load_clean_candidates()

    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], width_ratios=[1.05, 1.0],
                          hspace=0.6, wspace=0.95, left=0.32, right=0.90, top=0.90, bottom=0.09)
    axA = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, :])

    # ── Panel A: per-target CV ROC-AUC ───────────────────────────────────────
    y = np.arange(len(metrics))
    colors = [C["amber"] if e else C["mlp"] for e in metrics["exploratory"]]
    axA.barh(y, metrics["cv_roc_auc"], color=colors, height=0.72, zorder=3)
    axA.axvline(0.5, color=C["neutral"], lw=0.6, ls=":", zorder=1)
    axA.set_yticks(y)
    axA.set_yticklabels([f"{_short(t)} (n={n})" for t, n in
                         zip(metrics["target"], metrics["n_mols"])], fontsize=5.3)
    axA.set_xlim(0.4, 1.0); axA.set_xlabel("Cross-validated ROC-AUC")
    axA.set_xticks([0.4, 0.6, 0.8, 1.0])
    for yi, auc in enumerate(metrics["cv_roc_auc"]):
        axA.text(auc - 0.01, yi, f"{auc:.2f}", va="center", ha="right",
                 fontsize=4.8, color="white", fontweight="bold")
    leg = [mpatches.Patch(color=C["mlp"], label="n ≥ 100"),
           mpatches.Patch(color=C["amber"], label="n < 100 (exploratory)")]
    axA.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, fontsize=5, columnspacing=1.0, handlelength=1.2)
    add_panel_label(axA, "A", x=-0.62, y=1.13)

    # ── Panel C: folate-pathway worked example ───────────────────────────────
    folate = cand[cand["predicted_target"].isin(
        ["Dihydrofolate reductase", "Thymidylate synthase"])].copy()
    folate = folate.sort_values("target_score", ascending=False).head(12)[::-1]
    yf = np.arange(len(folate))
    fcol = [C["mlp"] if t == "Dihydrofolate reductase" else C["rf"]
            for t in folate["predicted_target"]]
    axC.barh(yf, folate["target_score"], color=fcol, height=0.72, zorder=3)
    axC.set_yticks(yf)
    axC.set_yticklabels([d.title() for d in folate["drug_name"]], fontsize=5.2)
    axC.set_xlim(0, 1.0); axC.set_xlabel("Target-model score")
    axC.set_xticks([0, 0.5, 1.0])
    leg2 = [mpatches.Patch(color=C["mlp"], label="DHFR"),
            mpatches.Patch(color=C["rf"], label="Thymidylate synthase")]
    axC.legend(handles=leg2, loc="lower center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, fontsize=5, columnspacing=1.0, handlelength=1.2,
               title="Antifolate candidates → folate pathway", title_fontsize=6)
    add_panel_label(axC, "C", x=-0.62, y=1.16)

    # ── Panel B: candidate x target heatmap ──────────────────────────────────
    order = metrics.sort_values("cv_roc_auc", ascending=False)   # nicer column order
    slugs = order["slug"].tolist()
    labels = [_short(t, 22) for t in order["target"]]
    # merge candidate ensemble score + name onto the target screen matrix
    sc = screen.merge(cand[["chembl_id", "drug_name", "prob_ensemble", "predicted_target"]],
                      left_on="molecule_chembl_id", right_on="chembl_id", how="inner")
    # pick the strongest target-resolved candidates: top by their best target score
    # surface the real repurposing hits: target-resolved candidates by rank
    top = sc[sc["predicted_target"] != "none"].sort_values(
        "prob_ensemble", ascending=False).drop_duplicates("drug_name").head(18)
    M = top[slugs].to_numpy(dtype=float)
    im = axB.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axB.set_xticks(range(len(slugs)))
    axB.set_xticklabels(labels, rotation=45, ha="right", fontsize=5)
    axB.set_yticks(range(len(top)))
    axB.set_yticklabels([d.title() for d in top["drug_name"]], fontsize=5)
    axB.set_title("Predicted per-target activity of top target-resolved candidates",
                  fontsize=6.5, pad=4)
    cb = fig.colorbar(im, ax=axB, shrink=0.9, pad=0.015)
    cb.set_label("Target-model probability", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    add_panel_label(axB, "B", x=-0.32)

    print(f"  targets={len(metrics)}  folate_candidates={len(folate)}  heatmap_rows={len(top)}")
    save_fig(fig, "figure_6_target_resolved")


if __name__ == "__main__":
    main()
