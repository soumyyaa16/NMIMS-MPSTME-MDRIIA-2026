"""Visual scene builder (cosmetic only: every added geom has contype=conaffinity=0, so physics is unchanged).

    python src/scene_build.py --case c160 --nifti D:/robotics/brats_cases/BraTS20_Training_160

Creates cases/<id>/scene/ (hi-res curvature-shaded brain, cut-away variants, MRI slice textures, nav-screen image)
and models/_vis_<id>.xml with: translucent illustrative head + head clamp, operating-room set, MRI cut-away planes
through the target, entry/target/trajectory markers and camera presets.

HONESTY NOTES (also shown in the dossier): the brain surface is an intensity-threshold mask of the skull-stripped T1c scan,
shaded by its own sulcal depth (not a validated cortical surface); the head, clamp and room are illustrative;
lesion regions are the supplied expert annotation.
"""
import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage import measure
from skimage.filters import gaussian

import import_case as ic
from surgical_tremor_controller import ROOT, STANDOFF, APPROACH_DIR, PATH_LEN, load_plan

MODELS = ROOT / "models"
BASE_XML = MODELS / "surgical_7dof_robot.xml"
PANDA_XML = MODELS / "franka_emika_panda" / "panda_surgical.xml"
PANDA_VIS = MODELS / "franka_emika_panda" / "panda_visual.xml"
HI_KEEP = 0.85            # fraction of faces kept in the hi-res brain
SLICE_L = 0.10            # m  half-extent of each cut-away MRI quadrant
SLICE_N = 220             # px per quadrant side
FLIPV = True              # MuJoCo plane textures: image row 0 is at +local y (calibrated by render test)


# ---------------------------------------------------------------- paths / meta
def case_paths(case):
    if case is None:
        meshdir, scene = ROOT / "meshes_v2", ROOT / "cases" / "orig" / "scene"
    else:
        meshdir, scene = ROOT / "cases" / case / "meshes", ROOT / "cases" / case / "scene"
    scene.mkdir(parents=True, exist_ok=True)
    return meshdir, scene, json.loads((meshdir / "export_meta.json").read_text())


def find_nifti_dir(case, root):
    if root is None:
        return None
    root = Path(root)
    if list(root.glob("*t1ce.nii*")):
        return root
    cands = [root / f"BraTS20_Training_{case[1:]}", root / case] if case else []
    for c in cands:
        if c.exists() and list(c.glob("*t1ce.nii*")):
            return c
    return None


# ---------------------------------------------------------------- obj with texture coordinates
def write_obj_uv(path, v, f, u):
    lines = [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in v]
    lines += [f"vt {x:.4f} 0.5" for x in u]
    lines += [f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}" for a, b, c in f]
    Path(path).write_text("\n".join(lines) + "\n")


def write_ramp(path):
    """Colour ramp: u=0 gyral crest (pale pink) -> u=1 deep sulcus (dark rose-grey)."""
    t = np.linspace(0, 1, 256)[:, None]
    pale, mid, dark = np.array([0.93, 0.78, 0.72]), np.array([0.80, 0.55, 0.52]), np.array([0.42, 0.24, 0.26])
    col = np.where(t < 0.5, pale + (mid - pale) * (t / 0.5), mid + (dark - mid) * ((t - 0.5) / 0.5))
    img = (np.repeat(col[None], 16, 0) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# ---------------------------------------------------------------- brain
def build_brain_hi(scene, meta, vol):
    if (scene / "brain_hi.obj").exists():
        return
    img, A, origin_mm, brain = vol["img"], vol["A"], np.array(meta["origin_mm_in_scanner_space"]), vol["brain"]
    flip = np.linalg.det(A[:3, :3]) < 0
    pad = 8
    sm = gaussian(np.pad(brain, pad).astype(float), 1.0)
    big = gaussian(np.pad(brain, pad).astype(float), 4.0)
    v, f, _, _ = measure.marching_cubes(sm, 0.5)
    sulc = map_coordinates(big, v.T, order=1) - 0.5               # > 0 : concave (sulcus), < 0 : convex (gyrus)
    v -= pad
    world = (nib.affines.apply_affine(A, v) - origin_mm) * 1e-3
    if flip:
        f = f[:, ::-1]
    mesh = trimesh.Trimesh(world, f, process=False)
    trimesh.smoothing.filter_taubin(mesh, iterations=6)
    try:
        n_target = int(len(mesh.faces) * HI_KEEP)
        m2 = mesh.simplify_quadric_decimation(face_count=n_target)
        idx = cKDTree(mesh.vertices).query(m2.vertices)[1]
        mesh, sulc = m2, sulc[idx]
    except Exception as e:
        print("   decimation skipped:", e)
    s = np.clip(sulc / (np.percentile(np.abs(sulc), 97) + 1e-9), -1, 1)
    u = np.clip(0.5 + 0.5 * s, 0, 1)
    write_obj_uv(scene / "brain_hi.obj", np.asarray(mesh.vertices), np.asarray(mesh.faces), u)
    np.savez_compressed(scene / "brain_hi_cache.npz", v=np.asarray(mesh.vertices, np.float32),
                        f=np.asarray(mesh.faces, np.int32), u=u.astype(np.float32))
    print(f"   brain_hi: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")


def cut_brain(scene, target, sgn):
    """Brain with the octant {sgn_k (x_k - t_k) > 0 for all k} removed.  Cached per sign pattern."""
    name = scene / f"brain_cut_{''.join('p' if s > 0 else 'n' for s in sgn)}.obj"
    if name.exists():
        return name
    z = np.load(scene / "brain_hi_cache.npz")
    v, f, u = z["v"].astype(float), z["f"], z["u"]
    cen = v[f].mean(1)
    rel = (cen - target) * np.array(sgn)
    keep = ~np.all(rel > 0, axis=1)
    f2 = f[keep]
    used, inv = np.unique(f2, return_inverse=True)
    write_obj_uv(name, v[used], inv.reshape(-1, 3), u[used])
    return name


# ---------------------------------------------------------------- MRI slices
def load_volume(nii_dir):
    t1c = nib.load(ic.find(nii_dir, "*t1ce.nii*", "*t1c.nii*"))
    seg = nib.load(ic.find(nii_dir, "*seg.nii*"))
    img = np.asarray(t1c.dataobj).astype(np.float32)
    lab = np.rint(np.asarray(seg.dataobj)).astype(np.int16)
    lm, _ = ic.label_map(lab)
    pos = img[img > 0]
    lo, hi = np.percentile(pos, 1), np.percentile(pos, 99.5)
    return dict(img=img, lab=lab, lm=lm, A=t1c.affine, Ainv=np.linalg.inv(t1c.affine), lo=lo, hi=hi,
                brain=ic.brain_mask(img))


def sample_plane(vol, meta, center, xax, yax, L, N, alpha_outside=True):
    origin_mm = np.array(meta["origin_mm_in_scanner_space"])
    xs = (np.arange(N) + 0.5) / N * L - L / 2
    ys = (L / 2 - (np.arange(N) + 0.5) / N * L) if FLIPV else ((np.arange(N) + 0.5) / N * L - L / 2)
    X, Y = np.meshgrid(xs, ys)
    pts = center + X[..., None] * xax + Y[..., None] * yax
    mm = pts.reshape(-1, 3) * 1e3 + origin_mm
    vx = (vol["Ainv"][:3, :3] @ mm.T).T + vol["Ainv"][:3, 3]
    I = map_coordinates(vol["img"], vx.T, order=1, mode="constant").reshape(N, N)
    S = map_coordinates(vol["lab"].astype(np.float32), vx.T, order=0, mode="constant").reshape(N, N)
    g = np.clip((I - vol["lo"]) / (vol["hi"] - vol["lo"]), 0, 1) ** 0.85
    rgb = np.repeat(g[..., None], 3, 2)
    cols = {"edema": (0.30, 0.55, 0.98), "necrotic": (0.98, 0.82, 0.20), "tumor": (0.96, 0.20, 0.20)}
    for name in ("edema", "necrotic", "tumor"):
        lbl = vol["lm"].get(name)
        if lbl is None:
            continue
        m = np.rint(S) == lbl
        if not m.any():
            continue
        edge = m & ~ndi.binary_erosion(m, iterations=1)
        c = np.array(cols[name])
        rgb[m] = 0.62 * rgb[m] + 0.38 * c
        rgb[edge] = c
    a = np.where(I > 0, 255, 0 if alpha_outside else 255).astype(np.uint8)
    return np.dstack([(rgb * 255).astype(np.uint8), a])


def plane_frame(axis, sgn):
    """Plane through the target, bounded to the quadrant that borders the removed octant.
    axis = index of the plane normal; returns (xax, yax, normal) with x x y = normal pointing into the octant."""
    e = np.eye(3)
    a, b = [k for k in range(3) if k != axis]
    nrm = e[axis] * sgn[axis]
    xax = e[a] * sgn[a]
    yax = np.cross(nrm, xax)
    return xax, yax, nrm


def write_cap(path, rgba, center, xax, yax, L, M=110):
    """Flat textured mesh covering only the brain section of one MRI quadrant (no black rectangles)."""
    from scipy.ndimage import zoom, binary_dilation
    a = binary_dilation(rgba[..., 3] > 0, iterations=1).astype(np.float32)
    mask = zoom(a, M / a.shape[0], order=0) > 0.5
    ii, jj = np.mgrid[0:M + 1, 0:M + 1]
    xl, yl = -L / 2 + jj * L / M, L / 2 - ii * L / M
    P = center + xl[..., None] * xax + yl[..., None] * yax
    uv = np.stack([jj / M, 1 - ii / M], -1)
    idx = np.arange((M + 1) ** 2).reshape(M + 1, M + 1)
    ci, cj = np.nonzero(mask)
    tri = np.concatenate([np.stack([idx[ci, cj], idx[ci, cj + 1], idx[ci + 1, cj + 1]], 1),
                          np.stack([idx[ci, cj], idx[ci + 1, cj + 1], idx[ci + 1, cj]], 1)])
    tri = np.concatenate([tri, tri[:, ::-1]])                         # both windings (two-sided)
    used, inv = np.unique(tri, return_inverse=True)
    V, T = P.reshape(-1, 3)[used], uv.reshape(-1, 2)[used]
    lines = [f"v {a_:.6f} {b_:.6f} {c_:.6f}" for a_, b_, c_ in V] + [f"vt {u_:.5f} {v_:.5f}" for u_, v_ in T]
    t3 = inv.reshape(-1, 3) + 1
    lines += [f"f {a_}/{a_} {b_}/{b_} {c_}/{c_}" for a_, b_, c_ in t3]
    Path(path).write_text("\n".join(lines) + "\n")


def slice_textures(scene, meta, vol, target, sgn):
    tag = "".join("p" if s > 0 else "n" for s in sgn)
    out = {}
    for axis, nm in ((2, "axial"), (1, "coronal"), (0, "sagittal")):
        p = scene / f"mri_{nm}_{tag}.png"
        xax, yax, nrm = plane_frame(axis, sgn)
        center = target + (xax + yax) * SLICE_L / 2
        pm = p.with_suffix(".obj")
        if not p.exists() or not pm.exists():
            rgba = sample_plane(vol, meta, center, xax, yax, SLICE_L, SLICE_N)
            Image.fromarray(rgba).save(p)
            write_cap(pm, rgba, center, xax, yax, SLICE_L)
        out[nm] = (p, center, xax, yax, nrm)
    return out


def nav_screen(scene, meta, vol, target):
    p = scene / "nav_screen.png"
    if p.exists():
        return p
    N, L = 300, 0.20
    e = np.eye(3)
    panes = []
    for (a, b) in ((0, 1), (1, 2), (0, 2)):                       # axial(x,y), sagittal(y,z), coronal(x,z)
        im = sample_plane(vol, meta, target, e[a], e[b], L, N, alpha_outside=False)[..., :3].copy()
        c = N // 2
        im[c - 1:c + 2, :, :] = [255, 60, 60]
        im[:, c - 1:c + 2, :] = [255, 60, 60]
        panes.append(im)
    row = np.concatenate(panes, 1)
    bar = np.zeros((28, row.shape[1], 3), np.uint8)
    bar[:] = (14, 110, 120)
    Image.fromarray(np.concatenate([bar, row], 0)).save(p)
    return p


# ---------------------------------------------------------------- XML assembly
def f3(v):
    return " ".join(f"{x:.5f}" for x in v)


def build_head_mesh(path, c, hx, hy, hz, z_bottom):
    """Illustrative human head: cranium, face (brow, eye sockets, nose, lips, chin), ears, neck, plus a separate hair shell.
    Built as a smooth union of ellipsoids (signed distance field) and meshed by marching cubes. Writes head + hair .obj."""
    path = Path(path)
    hair_path = path.with_name(path.stem.replace("head", "hair") + ".obj")
    if path.exists() and hair_path.exists():
        return
    step = 0.002
    xs = np.arange(c[0] - hx - 0.045, c[0] + hx + 0.045, step)
    ys = np.arange(c[1] - hy - 0.04, c[1] + hy + 0.07, step)
    zs = np.arange(z_bottom, c[2] + hz + 0.03, step)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X, Y, Z], -1).astype(np.float32)

    def ell(ctr, r):
        q = (P - np.array(ctr, np.float32)) / np.array(r, np.float32)
        return (np.linalg.norm(q, axis=-1) - 1.0) * min(r)

    def smin(a, b, k):
        h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
        return b * (1 - h) + a * h - k * h * (1 - h)

    def smax(a, b, k):
        return -smin(-a, -b, k)

    cx, cy, cz = c
    d = ell(c, (hx, hy, hz))                                                                    # cranium
    d = smin(d, ell((cx, cy + hy * 0.50, cz - hz * 0.55), (hx * 0.80, hy * 0.58, hz * 0.62)), 0.025)   # face mass (cheeks, forehead base)
    d = smin(d, ell((cx, cy + hy * 0.40, cz - hz * 1.05), (hx * 0.52, hy * 0.52, hz * 0.34)), 0.02)    # jaw and chin
    d = smin(d, ell((cx, cy + hy * 0.93, cz - hz * 0.14), (hx * 0.66, 0.017, 0.012)), 0.012)           # brow ridge
    d = smin(d, ell((cx, cy + hy * 0.98, cz - hz * 0.48), (0.0105, 0.030, 0.028)), 0.010)              # nose bridge
    d = smin(d, ell((cx, cy + hy * 1.06, cz - hz * 0.66), (0.013, 0.014, 0.011)), 0.008)               # nose tip
    d = smin(d, ell((cx, cy + hy * 0.95, cz - hz * 0.93), (0.021, 0.010, 0.006)), 0.006)               # lips
    for sx in (-1, 1):
        d = smin(d, ell((cx + sx * hx * 0.99, cy - hy * 0.06, cz - hz * 0.28), (0.007, 0.014, 0.024)), 0.006)   # ears
        sock = ell((cx + sx * hx * 0.36, cy + hy * 0.93, cz - hz * 0.30), (0.014, 0.012, 0.009))
        d = smax(d, -sock, 0.006)                                                                       # eye sockets
    skull_only = d.copy()
    nr = 0.048
    neck_top = cz - hz * 0.78
    cyl = np.maximum(np.hypot(X - cx, (Y - (cy - 0.012)) * 1.12) - nr, np.maximum(z_bottom - Z, Z - neck_top))
    d = smin(d, cyl, 0.03)
    v, f, _, _ = measure.marching_cubes(d, 0.0, spacing=(step, step, step))
    v += np.array([xs[0], ys[0], zs[0]])
    m = trimesh.Trimesh(v, f)
    trimesh.smoothing.filter_taubin(m, iterations=6)
    m.export(path)
    # hair: the upper part of the cranium, slightly thicker than the skin, with a hairline that is high at the forehead
    n = np.array([0.0, -0.45, 1.0], np.float32)
    n /= np.linalg.norm(n)
    p0 = np.array([cx, cy + hy * 0.15, cz + hz * 0.12], np.float32)
    above = (P - p0) @ n                                                    # > 0 above the hairline plane
    hair = np.maximum(skull_only - 0.0035, -above)
    v, f, _, _ = measure.marching_cubes(hair, 0.0, spacing=(step, step, step))
    v += np.array([xs[0], ys[0], zs[0]])
    mh = trimesh.Trimesh(v, f)
    trimesh.smoothing.filter_taubin(mh, iterations=5)
    mh.export(hair_path)


def build_body_mesh(path, c):
    """Illustrative reclined patient body (gown) from capsules, smooth-unioned and meshed."""
    if Path(path).exists():
        return
    step = 0.006
    xs = np.arange(-0.42, 0.42, step)
    ys = np.arange(c[1] - 0.30, c[1] + 0.64, step)
    zs = np.arange(-1.22, -0.05, step)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X, Y, Z], -1).astype(np.float32)

    def cap(a, b, r):
        a, b = np.array(a, np.float32), np.array(b, np.float32)
        pa, ba = P - a, b - a
        h = np.clip((pa @ ba) / (ba @ ba), 0, 1)[..., None]
        return np.linalg.norm(pa - ba * h, axis=-1) - r

    def smin(a, b, k):
        h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
        return b * (1 - h) + a * h - k * h * (1 - h)

    cy, cx = c[1], c[0]
    hip = (cx, cy - 0.12, -0.66)
    d = cap((cx, cy - 0.03, -0.27), hip, 0.125)                                    # torso (top below the neck base)
    d = smin(d, cap((cx - 0.19, cy - 0.04, -0.23), (cx + 0.19, cy - 0.04, -0.23), 0.058), 0.03)   # shoulders
    d = smin(d, cap((cx - 0.08, hip[1], -0.66), (cx + 0.08, hip[1], -0.66), 0.115), 0.03)         # pelvis
    for sx in (-1, 1):
        sh = (cx + sx * 0.20, cy - 0.04, -0.23)
        el = (cx + sx * 0.26, cy + 0.03, -0.46)
        wr = (cx + sx * 0.19, cy + 0.17, -0.62)
        d = smin(d, cap(sh, el, 0.044), 0.02)
        d = smin(d, cap(el, wr, 0.037), 0.02)
        kn = (cx + sx * 0.10, hip[1] + 0.42, -0.62)
        an = (cx + sx * 0.10, hip[1] + 0.50, -1.12)
        d = smin(d, cap((cx + sx * 0.09, hip[1], -0.66), kn, 0.078), 0.03)
        d = smin(d, cap(kn, an, 0.056), 0.03)
        d = smin(d, cap(an, (cx + sx * 0.10, an[1] + 0.13, -1.17), 0.045), 0.02)
    v, f, _, _ = measure.marching_cubes(d, 0.0, spacing=(step, step, step))
    v += np.array([xs[0], ys[0], zs[0]])
    m = trimesh.Trimesh(v, f)
    trimesh.smoothing.filter_taubin(m, iterations=6)
    m.export(path)


def bed_set(scene, A, W, c, hx, hy, hz, gh):
    """Reclined patient on a bed, head clamp, robot trolley, monitor cart, room.  Returns animation geom names."""
    build_body_mesh(scene / "body2.obj", c)
    A.append(f'<mesh name="body_m" file="{(scene / "body2.obj").resolve().as_posix()}"/>')
    A.append('<material name="gown" rgba="0.62 0.78 0.90 1" specular="0.08" shininess="0.15"/>')
    A.append('<material name="mattress" rgba="0.90 0.92 0.94 1" specular="0.05" shininess="0.1"/>')
    A.append('<material name="accent" rgba="0.05 0.55 0.58 1" specular="0.3" shininess="0.4"/>')
    A.append('<material name="rubber" rgba="0.08 0.08 0.09 1" specular="0.05" shininess="0.1"/>')
    z_floor = -1.30
    W.append(f'<geom name="body" type="mesh" mesh="body_m" material="gown" contype="0" conaffinity="0"/>')
    for sx in (-1, 1):
        W.append(f'<geom name="hand{sx}" type="sphere" pos="{f3([c[0] + sx*0.19, c[1] + 0.185, -0.635])}" size="0.036" material="skin_f" {gh}/>')
    # bed: backrest, seat, column, base
    tilt = float(np.arctan2(0.12, 0.45))
    W.append(f'<geom name="backrest" type="box" pos="{f3([c[0], c[1] - 0.20, -0.45])}" size="0.30 0.035 0.36" euler="{tilt:.4f} 0 0" material="mattress" {gh}/>')
    W.append(f'<geom name="seat" type="box" pos="{f3([c[0], c[1] + 0.22, -0.745])}" size="0.31 0.42 0.045" material="mattress" {gh}/>')
    W.append(f'<geom name="bed_frame" type="box" pos="{f3([c[0], c[1] + 0.12, -0.80])}" size="0.28 0.50 0.012" material="steel" {gh}/>')
    W.append(f'<geom name="bed_col" type="cylinder" pos="{f3([c[0], c[1] + 0.12, -1.04])}" size="0.06 0.24" material="steel" {gh}/>')
    W.append(f'<geom name="bed_base" type="box" pos="{f3([c[0], c[1] + 0.12, z_floor + 0.04])}" size="0.30 0.55 0.04" material="steel" {gh}/>')
    # head clamp: headrest board + two posts + lateral pins
    pz = c[2] - 0.028
    W.append(f'<geom name="headboard" type="box" pos="{f3([c[0], c[1] - 0.075, -0.20])}" size="{hx + 0.06:.3f} 0.012 0.045" material="steel" {gh}/>')
    for sx in (-1, 1):
        px = c[0] + sx * (hx + 0.045)
        W.append(f'<geom name="post{sx}" type="capsule" fromto="{f3([px, c[1]-0.07, -0.20])} {f3([px, c[1]-0.03, pz])}" size="0.011" material="steel" {gh}/>')
        W.append(f'<geom name="pin{sx}" type="capsule" fromto="{f3([px, c[1]-0.03, pz])} {f3([c[0] + sx*(hx*0.97), c[1]-0.03, pz])}" size="0.006" material="steel" {gh}/>')
    # room
    W.append(f'<geom name="floor" type="plane" pos="0 0 {z_floor}" size="6 6 0.05" material="floor_m" contype="0" conaffinity="0"/>')
    W.append(f'<geom name="wall_back" type="box" pos="0 2.6 0.2" size="5 0.02 1.7" material="wall_m" contype="0" conaffinity="0"/>')
    W.append(f'<geom name="wall_left" type="box" pos="-3.2 0 0.2" size="0.02 5 1.7" material="wall_m" contype="0" conaffinity="0"/>')
    for i, (lx, ly) in enumerate(((-0.10, -0.05), (0.35, 0.30))):
        W.append(f'<geom name="lamp{i}" type="cylinder" pos="{lx} {ly} 1.12" size="0.22 0.012" material="lamp_m" contype="0" conaffinity="0"/>')
        W.append(f'<geom name="lamp_arm{i}" type="cylinder" pos="{lx} {ly} 1.36" size="0.016 0.24" material="steel" contype="0" conaffinity="0"/>')
    # robot trolley (animated): column, base, wheels, top plate
    cart = []
    rx, ry = -0.45, 0.0
    def add(name, xml):
        W.append(xml); cart.append(name)
    add("cart_col", f'<geom name="cart_col" type="box" pos="{rx} {ry} -0.72" size="0.19 0.19 0.50" material="white_m" contype="0" conaffinity="0"/>')
    add("cart_stripe", f'<geom name="cart_stripe" type="box" pos="{rx} {ry-0.1905} -0.55" size="0.15 0.002 0.05" material="accent" contype="0" conaffinity="0"/>')
    add("cart_base", f'<geom name="cart_base" type="box" pos="{rx} {ry} {z_floor+0.09}" size="0.30 0.30 0.05" material="steel" contype="0" conaffinity="0"/>')
    add("cart_top", f'<geom name="cart_top" type="cylinder" pos="{rx} {ry} -0.1535" size="0.17 0.006" material="white_m" contype="0" conaffinity="0"/>')
    for k, (wx, wy) in enumerate(((-0.24, -0.24), (0.24, -0.24), (-0.24, 0.24), (0.24, 0.24))):
        add(f"cart_w{k}", f'<geom name="cart_w{k}" type="cylinder" pos="{rx+wx} {ry+wy} {z_floor+0.05}" size="0.05 0.025" euler="0 1.5708 0" material="rubber" contype="0" conaffinity="0"/>')
    # navigation monitor cart
    W.append('<geom name="mon_cart" type="box" pos="0.85 -0.40 -0.78" size="0.20 0.16 0.52" material="white_m" contype="0" conaffinity="0"/>')
    W.append('<geom name="mon_stem" type="cylinder" pos="0.85 -0.40 -0.12" size="0.016 0.16" material="steel" contype="0" conaffinity="0"/>')
    W.append('<geom name="mon_body" type="box" pos="0.85 -0.40 0.12" size="0.22 0.014 0.13" material="darkpad" contype="0" conaffinity="0"/>')
    return cart


def make_panda_visual():
    xml = PANDA_XML.read_text()
    tag = '<site name="needle_tip" pos="0 0 0.19" size="0.0015" rgba="1 0.1 0.1 1"/>'
    assert tag in xml, "panda_surgical.xml has an unexpected needle definition"
    cam = tag + '\n                      <camera name="tipcam" pos="0 0 0.188" euler="3.14159265 0 0" fovy="50"/>'
    PANDA_VIS.write_text(xml.replace(tag, cam))


def build_xml(case=None, plan=None, nifti=None, cutaway=True, name=None):
    meshdir, scene, meta = case_paths(case)
    vol = None
    nd = find_nifti_dir(case or "", nifti)
    if nd is not None:
        vol = load_volume(nd)
    target = np.array(meta["target_m"])
    if plan is None:
        plan = load_plan() if case is None else dict(
            target=target, entry=target + np.array([0.0, 0.0, 0.05]), standoff=target + np.array([0.0, 0.0, 0.11]),
            ins=np.array([0.0, 0.0, -1.0]))
    target = np.asarray(plan["target"], float)
    d_app = np.asarray(plan["standoff"], float) - target
    d_app = d_app / np.linalg.norm(d_app)
    sgn = [1 if x >= 0 else -1 for x in d_app]

    if vol is not None:
        build_brain_hi(scene, meta, vol)
    have_hi = (scene / "brain_hi.obj").exists()
    write_ramp(scene / "ramp.png")
    make_panda_visual()

    # brain extent -> head size
    if have_hi:
        z = np.load(scene / "brain_hi_cache.npz")
        bv = z["v"]
    else:
        bv = np.asarray(trimesh.load(meshdir / "brain.obj", process=False).vertices)
    lo, hi = bv.min(0), bv.max(0)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    hx, hy, hz = h[0] + 0.016, h[1] + 0.018, h[2] + 0.014
    z_table = -0.15

    A = []                                                     # assets
    W = []                                                     # worldbody
    ab = scene.resolve().as_posix()
    A.append(f'<texture name="ramp" type="2d" file="ramp.png"/>')
    A.append('<material name="cortex" texture="ramp" specular="0.55" shininess="0.75" reflectance="0.04"/>')
    if have_hi:
        A.append(f'<mesh name="brain_full" file="{ab}/brain_hi.obj"/>')
        if cutaway:
            cp = cut_brain(scene, target, sgn)
            A.append(f'<mesh name="brain_cut" file="{Path(cp).resolve().as_posix()}"/>')
    else:
        A.append(f'<mesh name="brain_full" file="{(meshdir / "brain.obj").resolve().as_posix()}"/>')
    for nm in ("tumor", "necrotic", "edema"):
        if (meshdir / f"{nm}.obj").exists():
            A.append(f'<mesh name="{nm}" file="{(meshdir / (nm + ".obj")).resolve().as_posix()}"/>')

    # brain geoms (group 1 = full, group 2 = cut-away)
    if cutaway and have_hi:
        W.append('<geom name="brain_geom" type="mesh" mesh="brain_cut" material="cortex" group="0" contype="0" conaffinity="0"/>')
        W.append('<geom name="brain_full_geom" type="mesh" mesh="brain_full" material="cortex" group="4" contype="0" conaffinity="0"/>')
    else:
        W.append('<geom name="brain_geom" type="mesh" mesh="brain_full" material="cortex" group="0" contype="0" conaffinity="0"/>')
    A.append('<material name="edema_m" rgba="0.30 0.55 0.98 0.30" specular="0.2" shininess="0.3"/>')
    A.append('<material name="necrotic_m" rgba="0.98 0.82 0.20 1" specular="0.3" shininess="0.4"/>')
    A.append('<material name="tumor_m" rgba="0.96 0.18 0.18 1" specular="0.5" shininess="0.6"/>')
    for nm in ("edema", "necrotic", "tumor"):
        if (meshdir / f"{nm}.obj").exists():
            W.append(f'<geom name="{nm}_geom" type="mesh" mesh="{nm}" material="{nm}_m" group="0" contype="0" conaffinity="0"/>')

    # MRI cut-away planes
    if cutaway and vol is not None:
        sl = slice_textures(scene, meta, vol, target, sgn)
        for nm, (p, cen, xax, yax, nrm) in sl.items():
            A.append(f'<texture name="tex_{nm}" type="2d" file="{p.name}"/>')
            A.append(f'<material name="mat_{nm}" texture="tex_{nm}" emission="0.30" specular="0" shininess="0" reflectance="0"/>')
            A.append(f'<mesh name="cap_{nm}" file="{p.with_suffix(".obj").resolve().as_posix()}" inertia="shell"/>')
            W.append(f'<geom name="mri_{nm}" type="mesh" mesh="cap_{nm}" material="mat_{nm}" group="0" contype="0" conaffinity="0"/>')
    # navigation-screen prop (static, illustrative)
    nav_tex = ""
    if vol is not None:
        nav = nav_screen(scene, meta, vol, target)
        A.append(f'<texture name="tex_nav" type="2d" file="{nav.name}"/>')
        A.append('<material name="mat_nav" texture="tex_nav" emission="0.7"/>')
        nav_tex = "mat_nav"

    # head (illustrative), clamp, table, drape
    A += ['<material name="skin_f" rgba="0.88 0.64 0.52 0.55" specular="0.25" shininess="0.35"/>',
          '<material name="skin" rgba="0.90 0.68 0.56 0.30" specular="0.30" shininess="0.40"/>',
          '<material name="drape_m" rgba="0.20 0.36 0.52 1" specular="0.03" shininess="0.1"/>',
          '<material name="steel" rgba="0.70 0.72 0.76 1" specular="0.7" shininess="0.7" reflectance="0.12"/>',
          '<material name="darkpad" rgba="0.20 0.24 0.30 1" specular="0.1" shininess="0.2"/>',
          '<material name="wall_m" rgba="0.60 0.73 0.83 1" specular="0.05" shininess="0.1"/>',
          '<material name="lamp_m" rgba="1 1 1 1" emission="1.0"/>',
          '<material name="floor_m" rgba="0.80 0.84 0.88 1" specular="0.3" shininess="0.4" reflectance="0.18"/>',
          '<material name="white_m" rgba="0.93 0.94 0.96 1" specular="0.3" shininess="0.4"/>',
          '<material name="mark_entry" rgba="1 0.85 0.1 1" emission="0.6"/>',
          '<material name="mark_target" rgba="0.15 1 0.45 1" emission="0.7"/>',
          '<material name="mark_path" rgba="0.2 0.9 1.0 0.85" emission="0.6"/>']
    gh = "group=\"0\" contype=\"0\" conaffinity=\"0\""
    build_head_mesh(scene / "head2.obj", c, hx, hy, hz, z_table)
    A.append(f'<mesh name="head_m" file="{(scene / "head2.obj").resolve().as_posix()}"/>')
    A.append(f'<mesh name="hair_m" file="{(scene / "hair2.obj").resolve().as_posix()}"/>')
    A.append('<material name="hair" rgba="0.16 0.11 0.08 1" specular="0.15" shininess="0.3"/>')
    A.append('<material name="eye" rgba="0.10 0.07 0.06 1" specular="0.6" shininess="0.8"/>')
    W.append(f'<geom name="head" type="mesh" mesh="head_m" material="skin" {gh}/>')
    W.append(f'<geom name="hair" type="mesh" mesh="hair_m" material="hair" {gh}/>')
    for sx in (-1, 1):
        W.append(f'<geom name="eye{sx}" type="sphere" pos="{f3([c[0] + sx * hx * 0.36, c[1] + hy * 0.90, c[2] - hz * 0.30])}" '
                 f'size="0.0065" material="eye" {gh}/>')
    cart_geoms = bed_set(scene, A, W, c, hx, hy, hz, gh)
    if nav_tex:
        W.append(f'<geom name="mon_screen" type="plane" pos="0.85 -0.3855 0.12" xyaxes="-1 0 0 0 0 1" size="0.205 0.115 0.001" material="{nav_tex}" contype="0" conaffinity="0"/>')

    # markers from the plan
    entry, stand = np.asarray(plan["entry"], float), np.asarray(plan["standoff"], float)
    W.append(f'<body name="target_body" pos="{f3(target)}">')
    W.append('  <site name="target_site" pos="0 0 0" size="0.0025" rgba="0.1 1 0.4 1"/>')
    W.append('  <geom name="tgt_ring" type="sphere" size="0.0032" material="mark_target" contype="0" conaffinity="0"/>')
    for ax in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        a = np.array(ax) * 0.0085
        W.append(f'  <geom type="capsule" fromto="{f3(-a)} {f3(a)}" size="0.0004" material="mark_target" contype="0" conaffinity="0"/>')
    W.append('</body>')
    W.append(f'<geom name="entry_mark" type="sphere" pos="{f3(entry)}" size="0.0032" material="mark_entry" contype="0" conaffinity="0"/>')
    W.append(f'<geom name="traj_line" type="capsule" fromto="{f3(target)} {f3(stand)}" size="0.00035" material="mark_path" contype="0" conaffinity="0"/>')

    cams = []
    sx_, sy_, sz_ = sgn
    for nm, off, fov in (("octant", (sx_*0.30, sy_*0.30, 0.17), 38), ("octant_close", (sx_*0.17, sy_*0.17, 0.10), 32),
                         ("octant_top", (sx_*0.05, sy_*0.12, 0.34), 38)):
        cams.append(f'<camera name="{nm}" mode="targetbody" target="target_body" pos="{f3(target + np.array(off))}" fovy="{fov}"/>')
    for nm, pos, fov in (("overview", (-0.55, -0.95, 0.55), 45), ("hero", (-0.42, -0.58, 0.30), 38),
                         ("left", (-0.95, 0.02, 0.12), 38), ("right", (0.95, 0.05, 0.12), 38),
                         ("top", (0.02, -0.12, 1.00), 38), ("front", (0.02, 0.95, 0.10), 38),
                         ("close", (-0.20, -0.26, 0.16), 32), ("room", (0.95, -1.35, 0.60), 55)):
        cams.append(f'<camera name="{nm}" mode="targetbody" target="target_body" pos="{f3(pos)}" fovy="{fov}"/>')

    lights = ['<light name="spot_key" type="spot" pos="-0.15 -0.10 1.10" dir="0.12 0.08 -1" cutoff="38" exponent="6" '
              'diffuse="1.0 0.98 0.94" specular="0.5 0.5 0.5" castshadow="true"/>',
              '<light name="spot_fill" type="spot" pos="0.25 0.20 1.10" dir="-0.2 -0.2 -1" cutoff="45" exponent="5" '
              'diffuse="0.55 0.58 0.64" specular="0.2 0.2 0.2" castshadow="false"/>',
              '<light name="rim" type="directional" pos="0 0 2" dir="0.5 -0.7 -0.5" diffuse="0.30 0.34 0.40" castshadow="false"/>']

    xml = f'''<mujoco model="surgical_7dof_robot_visual">
  <compiler angle="radian" meshdir="{ab}" texturedir="{ab}" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.42 0.44 0.48" diffuse="0.25 0.26 0.28" specular="0.05 0.05 0.05"/>
    <rgba haze="0.62 0.74 0.84 1"/>
    <global azimuth="145" elevation="-22" offwidth="1920" offheight="1080"/>
    <quality shadowsize="8192"/>
    <map shadowscale="0.5" znear="0.02"/>
  </visual>
  <asset>
    <model name="panda" file="franka_emika_panda/panda_visual.xml"/>
    <texture type="skybox" builtin="gradient" rgb1="0.80 0.88 0.95" rgb2="0.50 0.62 0.74" width="512" height="3072"/>
    {chr(10).join("    " + a for a in A)}
  </asset>
  <worldbody>
    {chr(10).join("    " + l for l in lights)}
    {chr(10).join("    " + w for w in W)}
    {chr(10).join("    " + c_ for c_ in cams)}
    <frame name="robot_mount" pos="-0.45 0 -0.15">
      <attach model="panda" body="link0" prefix="p_"/>
    </frame>
  </worldbody>
</mujoco>
'''
    out = MODELS / f"_vis_{case or 'orig'}.xml"
    out.write_text(xml)
    return out, dict(sgn=sgn, head_center=c.tolist(), head_half=[hx, hy, hz], cart_geoms=cart_geoms)


def load_visual(case=None, plan=None, nifti=None, cutaway=True):
    import mujoco
    path, info = build_xml(case, plan, nifti, cutaway)
    m = mujoco.MjModel.from_xml_path(str(path))
    return m, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--nifti", default=None, help="folder with BraTS cases (or one case folder)")
    ap.add_argument("--no-cutaway", action="store_true")
    a = ap.parse_args()
    p, info = build_xml(a.case, None, a.nifti, not a.no_cutaway)
    print("wrote", p, info)