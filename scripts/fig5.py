"""Figure 5 — SHAP feature importance (top ECFP4 bits, RF, probability output)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *  # noqa: F403,E402
import shap  # noqa: E402


def main():
    m = load_model_eval()
    rf, X_train, X_test = m["rf"], m["X_train"], m["X_test"]

    X_train_np = np.asarray(X_train, dtype=np.float64)
    X_test_np = np.asarray(X_test, dtype=np.float64)

    rng = np.random.default_rng(SEED)
    background = X_train_np[rng.choice(X_train_np.shape[0], size=200, replace=False)]
    X_explain = X_test_np[rng.choice(X_test_np.shape[0], size=500, replace=False)]

    print("  Computing SHAP (interventional, probability output)...")
    explainer = shap.TreeExplainer(rf, data=background,
                                   feature_perturbation="interventional",
                                   model_output="probability")
    shap_vals = explainer.shap_values(X_explain, check_additivity=False)

    if isinstance(shap_vals, list):
        sv = np.asarray(shap_vals[1])
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv = shap_vals[:, :, 1]
    else:
        sv = np.asarray(shap_vals)
    assert sv.shape == X_explain.shape, f"SHAP {sv.shape} vs X {X_explain.shape}"

    shap.summary_plot(sv, X_explain,
                      feature_names=[f"Bit {i:04d}" for i in range(X_train_np.shape[1])],
                      max_display=15, show=False, plot_size=(3.5, 5.0))

    fig = plt.gcf()
    for ax in fig.axes:
        ax.tick_params(labelsize=6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.axes[0].set_xlabel(
        "SHAP value (contribution to predicted antibacterial probability)", fontsize=7)
    plt.tight_layout(pad=0.3)
    save_fig(fig, "figure_5_shap_importance")


if __name__ == "__main__":
    main()
