"""Demo launcher:  python src\\launcher.py
Menu: 1) MuJoCo window  2) Video (render + play)  3) Play an existing video  4) Add a new MRI case  5) Doctor drives  Q) quit
Then: pick the case, pick the needle scenario (tumour part + approach side).
Cases live in  cases/<id>/  (meshes)  and  analytics/cases/<id>/scenarios.json  (planner output);
cases/registry.json remembers where each case's MRI files are, so every case you add stays playable.
Non-interactive:  python src\\launcher.py --mode video|live|drive --case c160 --scenario enhancing_superior [--quick]"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PY = sys.executable
DEFAULT_NIFTI_ROOT = r"D:\robotics\brats_cases"
REG = ROOT / "cases" / "registry.json"
FILMS = ROOT / "docs" / "film"

ZONE_TEXT = {"superior": "from the top", "anterior": "from the front", "posterior": "from the back",
             "left": "from the left side", "right": "from the right side", "inferior": "from below"}
LESION_TEXT = {"enhancing": "enhancing tumour (active edge)", "necrotic": "necrotic core (dead centre)",
               "edema": "oedema (swelling around)"}


def load_registry():
    try:
        return json.loads(REG.read_text())
    except Exception:
        return {}


def save_registry(reg):
    REG.parent.mkdir(parents=True, exist_ok=True)
    REG.write_text(json.dumps(reg, indent=2))


def list_cases():
    out = {}
    for f in sorted((ROOT / "analytics" / "cases").glob("*/scenarios.json")):
        try:
            out[f.parent.name] = json.loads(f.read_text())
        except Exception:
            pass
    return out


def nifti_for(case):
    reg = load_registry()
    if case in reg and Path(reg[case]["nifti"]).exists():
        return reg[case]["nifti"]
    return DEFAULT_NIFTI_ROOT


def ask(prompt, valid=None, default=None):
    while True:
        s = input(prompt).strip().strip('"')
        if not s and default is not None:
            return default
        if valid is None or s.lower() in valid:
            return s
        print("  please pick one of:", ", ".join(sorted(valid)))


def choose(title, items, allow_back=True):
    """items: list of (label). returns index or None for back."""
    print(f"\n{title}")
    for i, lab in enumerate(items, 1):
        print(f"  [{i}] {lab}")
    if allow_back:
        print("  [B] back")
    valid = {str(i) for i in range(1, len(items) + 1)} | ({"b"} if allow_back else set())
    s = ask("> ", valid)
    return None if s.lower() == "b" else int(s) - 1


def pick_case(cases):
    reg = load_registry()
    labels = []
    ids = list(cases)
    for c in ids:
        ok = sum(1 for s in cases[c] if s["status"] == "REACHABLE")
        tag = "  (added by you)" if c in reg else ""
        labels.append(f"{c}   {ok} reachable needle paths{tag}")
    i = choose("Which case (patient scan)?", labels)
    return None if i is None else ids[i]


def pick_scenario(case, scns):
    ok = [s for s in scns if s["status"] == "REACHABLE"]
    bad = [s for s in scns if s["status"] != "REACHABLE"]
    labels = [f"{LESION_TEXT.get(s['lesion'], s['lesion'])} - needle enters {ZONE_TEXT.get(s['zone'], s['zone'])}   [{s['id']}]" for s in ok]
    i = choose(f"Which needle path in {case}?", labels)
    if i is None:
        return None
    if bad:
        print(f"  (not offered: {len(bad)} paths the planner rejected, e.g. {bad[0]['id']})")
    return ok[i]["id"]


def run(cmd):
    print("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=False)
    except KeyboardInterrupt:
        print("\n(stopped)")


def play(mode, case, scenario, quick=False):
    if mode == "drive":
        print("Doctor-drives window opens. W/S move the needle, F = assistant on/off, H = help, Q = finish and save the log.")
        run([PY, str(SRC / "doctor_drives.py"), "--case", case, "--scenario", scenario, "--nifti", nifti_for(case)])
        return
    base = [PY, str(SRC / "make_film.py"), "--case", case, "--scenario", scenario, "--nifti", nifti_for(case)]
    if mode == "live":
        print("MuJoCo window opens (a second HUD window shows the live traces). Close the MuJoCo window when you're done.")
        run(base + ["--live"])
    else:
        out = FILMS / f"{case}_{scenario}.mp4"
        extra = ["--quick"] if quick else []
        print("Rendering - this takes " + ("1-2" if quick else "about 5") + " minutes; progress is printed. The film opens when it's done.")
        run(base + extra + ["--out", str(out)])


def play_existing():
    films = sorted(FILMS.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not films:
        print("No videos yet. Render one first (option 2).")
        return
    i = choose("Which video?", [f"{p.name}   ({datetime.datetime.fromtimestamp(p.stat().st_mtime):%d %b %H:%M})" for p in films])
    if i is not None:
        try:
            os.startfile(str(films[i]))
        except AttributeError:
            try:
                subprocess.Popen(["xdg-open", str(films[i])])
            except Exception as e:
                print("could not open it automatically:", e)


def add_case():
    print("\nAdd a new MRI case. You need a folder with a T1-contrast scan (*t1ce.nii*) and the lesion annotation (*seg.nii*), BraTS format.")
    folder = ask("Folder path (or Enter to cancel): ", default="")
    if not folder:
        return None
    src = Path(folder)
    if not src.is_dir():
        print("  not a folder.")
        return None
    has_t1 = list(src.glob("*t1ce.nii*")) or list(src.glob("*t1c.nii*"))
    has_seg = list(src.glob("*seg.nii*"))
    if not has_t1:
        print("  no *t1ce.nii* file found in that folder.")
        return None
    if not has_seg:
        print("  No lesion annotation (*seg.nii*) found.\n"
              "  An automatic segmentation model is NOT included yet (future work), so a case needs an expert\n"
              "  annotation like BraTS provides. Nothing was added.")
        return None
    m = re.search(r"(\d+)\D*$", src.name)
    default_id = "c" + (m.group(1)[-3:].zfill(3) if m else "900")
    cid = ask(f"Case id [{default_id}]: ", default=default_id)
    cid = re.sub(r"[^A-Za-z0-9_]", "", cid) or default_id
    if (ROOT / "analytics" / "cases" / cid / "scenarios.json").exists():
        if ask(f"  {cid} already exists. Replace it? [y/N]: ", default="n").lower() != "y":
            return None
    print("\nStep 1/2: building the brain and tumour meshes from the scan ...")
    r = subprocess.run([PY, str(SRC / "import_case.py"), "--src", str(src), "--id", cid], cwd=str(ROOT))
    if r.returncode != 0:
        print("  import failed (see messages above).")
        return None
    print("\nStep 2/2: the planner is checking every needle path (this takes a few minutes) ...")
    r = subprocess.run([PY, str(SRC / "plan_scenarios.py"), "--case", cid], cwd=str(ROOT))
    if r.returncode != 0:
        print("  planning failed (see messages above).")
        return None
    reg = load_registry()
    reg[cid] = {"nifti": str(src), "added": datetime.datetime.now().isoformat(timespec="minutes"), "source_folder": src.name}
    save_registry(reg)
    print(f"\nCase {cid} added and saved. It now appears in the case list.")
    return cid


def _open(path):
    try:
        os.startfile(str(path))
    except AttributeError:
        try:
            subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            print("could not open it automatically:", e)


def open_site():
    import webbrowser
    pages = [("Business model + results", "index.html"), ("Tremor Lab (live filter demo)", "lab.html"),
             ("Case Gallery (all MRI cases, videos)", "cases.html")]
    pages = [(lab, ROOT / "docs" / "site" / f) for lab, f in pages if (ROOT / "docs" / "site" / f).exists()]
    if not pages:
        print("No site pages found in docs/site.")
        return
    while True:
        i = choose("Which page?", [lab for lab, _ in pages])
        if i is None:
            return
        webbrowser.open(pages[i][1].as_uri())


def open_results():
    items = [("Filter comparison (all methods, tier by tier)", "docs/figs/compare_methods_kp50_10s.png"),
             ("Hold error box-plot, baseline vs learned filter", "docs/figs/section4_boxplot.png"),
             ("Sensor noise and delay sweep", "docs/figs/sensor_sweep.png"),
             ("Business: sensitivity", "docs/figures/fig_sensitivity.png"),
             ("Business: payback vs volume", "docs/figures/fig_volume.png"),
             ("Business: queue", "docs/figures/fig_queue.png"),
             ("Business: tornado", "docs/figures/fig_tornado.png")]
    items = [(lab, ROOT / rel) for lab, rel in items if (ROOT / rel).exists()]
    if not items:
        print("No result figures found yet.")
        return
    while True:
        i = choose("Which figure?", [lab for lab, _ in items])
        if i is None:
            return
        _open(items[i][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "video", "drive"])
    ap.add_argument("--case")
    ap.add_argument("--scenario")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.mode and a.case and a.scenario:
        play(a.mode, a.case, a.scenario, a.quick)
        return
    while True:
        print("\n==============  Surgical robot assistant - demo launcher  ==============")
        k = ask("  [1] MuJoCo window (live)\n  [2] Video (render and play)\n  [3] Play a video I already made\n"
                "  [4] Add a new MRI case\n  [5] Doctor drives (YOU move the needle, with beeps)\n"
                "  [6] Business model + results website\n  [7] Result figures (filters, sensors, business)\n  [Q] quit\n> ", {"1", "2", "3", "4", "5", "6", "7", "q"})
        if k.lower() == "q":
            return
        if k == "3":
            play_existing()
            continue
        if k == "6":
            open_site()
            continue
        if k == "7":
            open_results()
            continue
        if k == "4":
            cid = add_case()
            if cid and ask("Play it now? [y/N]: ", default="n").lower() == "y":
                cases = list_cases()
                sc_ = pick_scenario(cid, cases[cid])
                if sc_:
                    mode = {"1": "live", "2": "video", "3": "drive"}[ask("  [1] MuJoCo window  [2] Video  [3] Doctor drives: ", {"1", "2", "3"})]
                    play(mode, cid, sc_)
            continue
        mode = {"1": "live", "2": "video", "5": "drive"}[k]
        cases = list_cases()
        if not cases:
            print("No cases found in analytics/cases. Add one with option 4.")
            continue
        cid = pick_case(cases)
        if cid is None:
            continue
        scn = pick_scenario(cid, cases[cid])
        if scn is None:
            continue
        quick = False
        if mode == "video":
            quick = ask("Quality: [1] full (about 5 min)  [2] quick preview (1-2 min): ", {"1", "2"}) == "2"
        play(mode, cid, scn, quick)


if __name__ == "__main__":
    main()