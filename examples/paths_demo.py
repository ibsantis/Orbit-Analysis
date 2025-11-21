"""
Run with:
    PYTHONPATH=src python examples/paths_demo.py

Tip: To use local absolute paths, copy the template:
    cp src/orbit_analysis/config_template.py config_local.py  # then edit values
"""
from orbit_analysis.config import PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, SCRATCH_DIR, ensure_dirs

def main() -> None:
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_DIR    :", DATA_DIR)
    print("OUTPUT_DIR  :", OUTPUT_DIR)
    print("SCRATCH_DIR :", SCRATCH_DIR)
    ensure_dirs()
    out = OUTPUT_DIR / "paths_demo_ok.txt"
    out.write_text("ok\n")
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
