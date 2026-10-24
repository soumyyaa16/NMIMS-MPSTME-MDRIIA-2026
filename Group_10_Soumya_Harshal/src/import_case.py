"""Case importer: BraTS scan + lesion annotation  ->  meshes, metadata, planner-ready case folder.
    python src/import_case.py --src D:/robotics/brats_cases/BraTS20_Training_040 --id c040
    python src/import_case.py --all D:/robotics/brats_cases --select 6
Writes cases/<id>/meshes/{brain,tumor,necrotic,edema}.obj + export_meta.json and cases/summary.csv.
'tumor' = enhancing tumour.  Lesion regions come from the SUPPLIED annotation (expert BraTS labels);
the brain surface is an intensity-threshold mask of the T1c volume (BraTS is skull-stripped)."""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import trimesh
from scipy import ndimage as ndi
from skimage import measure
from skimage.filters import gaussian

ROOT = Path(__file__).resolve().parent.parent
SIGMAS = {"brain": 1.5, "tumor": 1.0, "necrotic": 1.0, "edema": 1.0}
DECIMATE = {"brain": 0.3, "tumor": None, "necrotic": None, "edema": None}


def find(src: Path, *patterns):
    for pat in patterns:
        hits = sorted(glob.glob(str(src / pat)))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"none of {patterns} in {src}")


def label_map(lab):
    """Per-dataset label scheme: BraTS<=2020 uses 4 for enhancing tumour, BraTS2021+ uses 3."""
    present = set(np.unique(lab).tolist())
    if 4 in present and 3 not in present:
        return {"necrotic": 1, "edema": 2, "tumor": 4}, "BraTS<=2020 (1=NCR/NET, 2=ED, 4=ET)"
    return {"necrotic": 1, "edema": 2, "tumor": 3}, "BraTS2021+ (1=NCR, 2=ED, 3=ET)"


def brain_mask(img):
    m = img > img[img > 0].mean() * 0.25
    cc, n = ndi.label(m)
    m = cc == (np.argmax(ndi.sum(m, cc, range(1, n + 1))) + 1)
    return ndi.binary_fill_holes(m)


def import_case(src: Path, case_id: str, out_root: Path, quiet=False):
    t0 = time.time()
    t1c = nib.load(find(src, "*t1ce.nii*", "*t1c.nii*", "*t1c.*"))
    seg = nib.load(find(src, "*seg.nii*"))
    A = t1c.affine
    assert np.allclose(A, seg.affine, atol=1e-3), "t1c and seg affines differ"
    img = np.asarray(t1c.dataobj).astype(np.float32)
    lab = np.rint(np.asarray(seg.dataobj)).astype(int)
    lm, scheme = label_map(lab)
    brain = brain_mask(img)
    masks = {"brain": brain, **{k: lab == v for k, v in lm.items()}}
    vox = np.sqrt((A[:3, :3] ** 2).sum(0))
    vox_cm3 = float(np.prod(vox)) * 1e-3
    origin_mm = nib.affines.apply_affine(A, np.argwhere(brain).mean(0))
    flip = np.linalg.det(A[:3, :3]) < 0

    def to_world_m(v):
        return (nib.affines.apply_affine(A, v) - origin_mm) * 1e-3

    def build(mask, sigma, decimate):
        pad = 8
        sm = gaussian(np.pad(mask, pad).astype(float), sigma)
        v, f, _, _ = measure.marching_cubes(sm, 0.5)
        v -= pad
        if flip:
            f = f[:, ::-1]
        mesh = trimesh.Trimesh(to_world_m(v), f, process=True)
        trimesh.smoothing.filter_taubin(mesh, iterations=10)
        if decimate:
            try:
                mesh = mesh.simplify_quadric_decimation(decimate)
            except Exception as e:
                if not quiet:
                    print("   decimation skipped:", e)
        mesh.fix_normals()
        return mesh

    meshdir = out_root / case_id / "meshes"
    meshdir.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, mk in masks.items():
        if mk.sum() < 50:                                  # label absent in this case
            info[name + "_mesh"] = "absent"
            continue
        m = build(mk, SIGMAS[name], DECIMATE[name])
        m.export(meshdir / f"{name}.obj")
        info[name + "_mesh"] = f"{len(m.vertices)}v watertight={m.is_watertight}"

    # target = deepest interior voxel of the enhancing region (falls back to necrotic, then edema)
    dist_brain = ndi.distance_transform_edt(brain, sampling=vox)
    target_label = next((n for n in ("tumor", "necrotic", "edema") if masks[n].sum() >= 50), None)
    if target_label is None:
        raise RuntimeError("no lesion label found in segmentation")
    dist = ndi.distance_transform_edt(masks[target_label], sampling=vox)
    tv = np.array(np.unravel_index(np.argmax(dist), dist.shape))
    target_m = to_world_m(tv)
    cc, ncomp = ndi.label(masks["tumor"]) if masks["tumor"].any() else (None, 0)
    sizes = ndi.sum(masks["tumor"], cc, range(1, ncomp + 1)) if ncomp else []
    n_et_blobs = int(sum(s > 100 for s in sizes))
    meta = dict(
        case_id=case_id, source=str(src), units="metres", world_axes="x right, y anterior, z superior (RAS)",
        origin="brain mask centroid", nifti_affine=A.tolist(), nifti_axcodes="".join(nib.aff2axcodes(A)),
        origin_mm_in_scanner_space=origin_mm.tolist(), label_scheme=scheme,
        target_m=target_m.tolist(), target_region=target_label, target_depth_mm=float(dist.max()),
        target_depth_below_brain_surface_mm=float(dist_brain[tuple(tv)]),
        brain_volume_cm3=float(brain.sum() * vox_cm3),
        vol_enhancing_cm3=float(masks["tumor"].sum() * vox_cm3), vol_necrotic_cm3=float(masks["necrotic"].sum() * vox_cm3),
        vol_edema_cm3=float(masks["edema"].sum() * vox_cm3), n_enhancing_blobs=n_et_blobs,
        mesh_info=info, annotation_note="lesion regions from supplied expert annotation, not a diagnosis")
    (meshdir / "export_meta.json").write_text(json.dumps(meta, indent=2))
    row = dict(case_id=case_id, scheme=scheme.split(" ")[0], target_region=target_label,
               vol_enh_cm3=meta["vol_enhancing_cm3"], vol_nec_cm3=meta["vol_necrotic_cm3"], vol_edema_cm3=meta["vol_edema_cm3"],
               inscribed_radius_mm=meta["target_depth_mm"], depth_below_surface_mm=meta["target_depth_below_brain_surface_mm"],
               tx_mm=target_m[0] * 1e3, ty_mm=target_m[1] * 1e3, tz_mm=target_m[2] * 1e3, n_enh_blobs=n_et_blobs,
               brain_cm3=meta["brain_volume_cm3"], seconds=time.time() - t0)
    if not quiet:
        print(f"{case_id}: {scheme.split(' ')[0]}  target {np.round(target_m*1e3,1)} mm in {target_label} "
              f"(r {row['inscribed_radius_mm']:.1f} mm, {row['depth_below_surface_mm']:.0f} mm below surface)  "
              f"ET {row['vol_enh_cm3']:.1f} / NEC {row['vol_nec_cm3']:.1f} / ED {row['vol_edema_cm3']:.1f} cm3  "
              f"[{row['seconds']:.0f}s]", flush=True)
    return row


def select_diverse(df, k):
    """Greedy farthest-point selection over standardised lesion features (not hand-picked)."""
    feats = np.c_[np.log1p(df.vol_enh_cm3), np.log1p(df.vol_edema_cm3), df.depth_below_surface_mm,
                  df.inscribed_radius_mm, df.tx_mm, df.ty_mm, df.tz_mm]
    Z = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    chosen = [int(np.argmax(np.linalg.norm(Z - Z.mean(0), axis=1)))]
    while len(chosen) < min(k, len(df)):
        d = np.min(np.linalg.norm(Z[:, None] - Z[chosen][None], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return [df.case_id.iloc[i] for i in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="one case folder containing *t1ce.nii and *seg.nii")
    ap.add_argument("--id", help="case id for --src (default: folder name)")
    ap.add_argument("--all", help="folder whose sub-folders are cases")
    ap.add_argument("--select", type=int, default=0, help="after --all, mark the k most different cases")
    ap.add_argument("--out", default=str(ROOT / "cases"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.src:
        import_case(Path(a.src), a.id or Path(a.src).name, out)
        return
    if not a.all:
        sys.exit("give --src or --all")
    rows = []
    for d in sorted(Path(a.all).iterdir()):
        if d.is_dir() and list(d.glob("*seg.nii*")):
            try:
                rows.append(import_case(d, "c" + d.name[-3:] if d.name.startswith("BraTS20_") else d.name, out))
            except Exception as e:
                print(f"{d.name}: FAILED {e}")
    df = pd.DataFrame(rows)
    if a.select:
        sel = select_diverse(df, a.select)
        df["selected"] = df.case_id.isin(sel)
        print("\nmost different cases:", sel)
    df.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()