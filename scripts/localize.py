# Run once after cloning:  python scripts/localize.py
# 1) config.py / session.py: project root from $BTX_ROOT (default: this repo) and dataset root from $BTX_DATA_ROOT.
# 2) notebooks/*.ipynb -> notebooks/*_local.ipynb : remove Kaggle-only cells, fix hard-coded paths, clear outputs.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OLD = '"/kaggle/working/BrainTumor_XAI_Project"'
NEW = '__import__("os").environ.get("BTX_ROOT", __import__("os").path.abspath(".."))'

(ROOT / "src/utils/config.py").write_text("\n".join([
    "import os", "from pathlib import Path",
    "PROJECT_ROOT = Path(os.environ.get('BTX_ROOT', Path(__file__).resolve().parents[2]))",
    "_KAGGLE = Path('/kaggle/input/datasets/briscdataset/brisc2025/brisc2025')",
    "DATA_ROOT = Path(os.environ.get('BTX_DATA_ROOT', _KAGGLE if _KAGGLE.exists() else PROJECT_ROOT / 'data' / 'raw' / 'brisc2025'))",
    "OUTPUT_DIR = PROJECT_ROOT / 'outputs'", "SEED = 42", ""]))

p = ROOT / "src/utils/session.py"; s = p.read_text()
for a, b in [("ROOT = Path(\"/kaggle/working/BrainTumor_XAI_Project\")",
              "import os\nROOT = Path(os.environ.get('BTX_ROOT', Path(__file__).resolve().parents[2]))"),
             ("assert fx == \"fc5ecbc245817db5\"", "assert os.environ.get('BTX_SKIP_FINGERPRINT') == '1' or fx == \"fc5ecbc245817db5\"")]:
    if a in s: s = s.replace(a, b)
    else: print("note: pattern not found (already patched?):", a[:40])
p.write_text(s)

try:
    import nbformat
except ImportError:
    raise SystemExit("pip install nbformat   (then rerun to convert the notebooks)")
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    if path.stem.endswith("_local"): continue
    nb = nbformat.read(path, as_version=4); keep = []
    for c in nb.cells:
        if c.cell_type == "code":
            if c.source.lstrip().startswith("%%writefile") or "/kaggle/input/*/BrainTumor_XAI_Project" in c.source: continue
            c.source = c.source.replace(OLD, NEW); c.outputs = []; c.execution_count = None
        keep.append(c)
    nb.cells = keep; nbformat.write(nb, path.with_name(path.stem + "_local.ipynb")); print("wrote", path.stem + "_local.ipynb")
