"""Render showcase videos for the web gallery and build docs/site/cases.json.
    python src\\batch_videos.py                  one video per case x lesion type (safest needle path), quick quality
    python src\\batch_videos.py --full           full quality (about 5 min each)
    python src\\batch_videos.py --all            EVERY reachable needle path (hours)
    python src\\batch_videos.py --only c200      only one case
    python src\\batch_videos.py --index-only     render nothing: copy the films you already have and rebuild cases.json
Videos that already exist in docs\\film are reused, so you can stop and re-run any time."""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launcher as L

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true")
ap.add_argument("--all", action="store_true")
ap.add_argument("--only")
ap.add_argument("--index-only", action="store_true")
a = ap.parse_args()

def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("Browsers cannot play the films as they are (MPEG-4 part 2). Install a converter once:  pip install imageio-ffmpeg   then re-run.")


def to_web(src, dst):
    """re-encode to H.264 so every browser can play it"""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)])
    if r.returncode != 0:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed on {src.name}")


FFMPEG = ffmpeg_exe()
SITE = L.ROOT / "docs" / "site"
VID = SITE / "videos"
VID.mkdir(parents=True, exist_ok=True)
L.FILMS.mkdir(parents=True, exist_ok=True)
cases = L.list_cases()
if a.only:
    cases = {k: v for k, v in cases.items() if k == a.only}


def pick(scns):
    ok = [s for s in scns if s["status"] == "REACHABLE"]
    if a.all:
        return ok
    best = {}
    for s in ok:                                    # safest path per lesion type = highest minimum manipulability
        if s["lesion"] not in best or s["min_w"] > best[s["lesion"]]["min_w"]:
            best[s["lesion"]] = s
    return [best[k] for k in ("enhancing", "necrotic", "edema") if k in best]


todo = [(c, s) for c, scns in cases.items() for s in pick(scns)]
print(f"{len(todo)} videos wanted ({'full' if a.full else 'quick'} quality)")
for n, (c, s) in enumerate(todo, 1):
    out = L.FILMS / f"{c}_{s['id']}.mp4"
    if out.exists() or a.index_only:
        print(f"[{n}/{len(todo)}] {out.name}: {'already there' if out.exists() else 'skipped (index-only)'}")
        continue
    cmd = [L.PY, str(L.SRC / "make_film.py"), "--case", c, "--scenario", s["id"], "--nifti", L.nifti_for(c)]
    cmd += [] if a.full else ["--quick"]
    cmd += ["--out", str(out)]
    print(f"[{n}/{len(todo)}] rendering {out.name} ...", flush=True)
    r = subprocess.run(cmd, cwd=str(L.ROOT))
    if r.returncode != 0 or not out.exists():
        print("   FAILED (continuing)")

index = []
for c, scns in cases.items():
    items = []
    for s in scns:
        src = L.FILMS / f"{c}_{s['id']}.mp4"
        if s["status"] != "REACHABLE" or not src.exists():
            continue
        to_web(src, VID / src.name)
        items.append(dict(id=s["id"], lesion=s["lesion"], lesion_text=L.LESION_TEXT.get(s["lesion"], s["lesion"]),
                          zone=s["zone"], zone_text=L.ZONE_TEXT.get(s["zone"], s["zone"]),
                          video="videos/" + src.name, size_mb=round((VID / src.name).stat().st_size / 1e6, 1),
                          path_mm=round(s["path_len_m"] * 1000, 1), clearance_mm=round(s["clearance_mm"], 1)))
    if items:
        index.append(dict(id=c, reachable=sum(1 for s in scns if s["status"] == "REACHABLE"), scenarios=items))
payload = dict(generated=datetime.datetime.now().isoformat(timespec="minutes"), cases=index)
(SITE / "cases.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
(SITE / "cases.js").write_text("window.CASES=" + json.dumps(payload) + ";\n", encoding="utf-8")   # works when the page is opened from disk
print(f"wrote {SITE / 'cases.js'} with {sum(len(c['scenarios']) for c in index)} videos across {len(index)} cases")
print("total video size: %.0f MB" % (sum(p.stat().st_size for p in VID.glob('*.mp4')) / 1e6))
