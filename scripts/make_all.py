"""Run every figure script in order. Usage: python figures/make_all.py"""
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULES = ["fig1", "fig2", "fig3", "fig4", "fig5",
           "figS1", "figS2", "figS3", "figS4", "figS5"]


def main():
    for name in MODULES:
        t0 = time.time()
        print(f"[{name}]")
        try:
            mod = importlib.import_module(name)
            mod.main()
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
        print(f"  ({time.time() - t0:.1f}s)")
    print("\nDone.")


if __name__ == "__main__":
    main()
