"""Build every figure in order.

Run:
    amr-figures        # or: python -m amr_repurposing.viz.figures.make_all
"""
import importlib
import time

MODULES = ["fig_workflow", "fig_model",
           "fig1", "fig2", "fig3", "fig4", "fig5", "fig_targets",
           "figS1", "figS2", "figS3", "figS4", "figS5"]

_PKG = "amr_repurposing.viz.figures"


def main():
    for name in MODULES:
        t0 = time.time()
        print(f"[{name}]")
        try:
            mod = importlib.import_module(f"{_PKG}.{name}")
            mod.main()
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
        print(f"  ({time.time() - t0:.1f}s)")
    print("\nDone.")


if __name__ == "__main__":
    main()
