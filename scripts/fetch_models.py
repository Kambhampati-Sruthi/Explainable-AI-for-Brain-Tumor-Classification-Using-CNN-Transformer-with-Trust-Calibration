# Download the model files of a GitHub Release and verify their SHA-256.
# Usage:  python scripts/fetch_models.py <github_user>/<repo> <tag>      e.g.  python scripts/fetch_models.py alice/brain-tumor-xai v1.0.0
import hashlib, sys, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
repo, tag = sys.argv[1], sys.argv[2]
dest = ROOT / "models" / "deploy"; dest.mkdir(parents=True, exist_ok=True)
for line in (dest / "SHA256SUMS.txt").read_text().splitlines():
    if not line.strip(): continue
    digest, name = line.split(maxsplit=1); name = name.strip().lstrip("*")
    target = dest / name
    if not target.exists():
        url = f"https://github.com/{repo}/releases/download/{tag}/{name}"; print("downloading", url); urllib.request.urlretrieve(url, target)
    ok = hashlib.sha256(target.read_bytes()).hexdigest() == digest
    print(("OK   " if ok else "BAD  ") + name)
    if not ok: sys.exit("checksum mismatch: " + name)
