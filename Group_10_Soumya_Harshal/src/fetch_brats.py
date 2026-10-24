"""Download selected BraTS2020 cases from Kaggle (single files, not the 42 GB archive).
    python src/fetch_brats.py --cases 1            # test with one case first
    python src/fetch_brats.py                      # default spread of 10 cases
Needs KAGGLE_API_TOKEN set (or kaggle.json) and `pip install kaggle`.
Data go to D:/robotics/brats_cases (outside the repo; do not commit BraTS data)."""
import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

SLUG = "awsaf49/brats20-dataset-training-validation"
BASE = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
DEFAULT = [1, 40, 80, 120, 160, 200, 240, 280, 320, 360]
MODS = ["t1ce", "seg", "flair", "t1", "t2"]


def collect(cdir: Path, fname: str) -> bool:
    """The Kaggle CLI recreates the remote folder tree and zips each file.  Find the file wherever it
    landed (zip or plain), put a plain copy directly in cdir, remove the leftovers."""
    target = cdir / fname
    for z in list(cdir.rglob(fname + ".zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.parent)
        z.unlink()
    for f in cdir.rglob(fname):
        if f != target:
            shutil.move(str(f), str(target))
    return target.exists() and target.stat().st_size > 1_000_000


def prune_empty(cdir: Path):
    for d in sorted([p for p in cdir.rglob("*") if p.is_dir()], key=lambda p: -len(p.parts)):
        if not any(d.iterdir()):
            d.rmdir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(map(str, DEFAULT)), help="comma list of case numbers 1..369")
    ap.add_argument("--out", default=r"D:\robotics\brats_cases")
    ap.add_argument("--mods", default=",".join(MODS))
    a = ap.parse_args()

    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise SystemExit("kaggle command not found - activate the .venv and run: python -m pip install kaggle")
    out = Path(a.out)
    for n in [int(x) for x in a.cases.split(",")]:
        cid = f"BraTS20_Training_{n:03d}"
        cdir = out / cid
        cdir.mkdir(parents=True, exist_ok=True)
        for mod in a.mods.split(","):
            fname = f"{cid}_{mod}.nii"
            if collect(cdir, fname):                     # already downloaded (possibly still zipped/nested)
                print("have", fname)
                continue
            print("downloading", fname, flush=True)
            r = subprocess.run([kaggle, "datasets", "download", "-d", SLUG, "-f", f"{BASE}/{cid}/{fname}",
                                "-p", str(cdir), "--force"], capture_output=True, text=True)
            if r.returncode != 0:
                print("  FAILED:", (r.stderr or r.stdout).strip()[:300])
                continue
            ok = collect(cdir, fname)
            print("  ok" if ok else "  WARNING: file not found after download")
        prune_empty(cdir)
    print("done ->", out)


if __name__ == "__main__":
    main()