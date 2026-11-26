"""Br35H data utilities: tumor masks (VIA annotations), scan clusters, leakage flags, brain mask.

Verified against the public release (github.com/AhmedHamada0/Brain-Tumor-Detection-Dataset):
  * 801 annotated images (TRAIN 500 / VAL 201 / TEST 100), 931 regions:
    666 polygons, 257 ellipses (143 rotated, theta in radians), 8 circles.
  * every annotated image is byte-identical (MD5) to yes/<same name>, so masks are
    matched to the classification set by MD5, which survives renaming and de-duplication.
  * images have 111 different native sizes -> masks are rasterised at native size and then
    resized exactly like the model input (A.Resize(224,224), no aspect preservation).
"""
import hashlib, json, os, glob
import numpy as np
from PIL import Image, ImageDraw


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _ellipse_poly(cx, cy, rx, ry, theta=0.0, n=72):
    # VIA draws with canvas ellipse(cx, cy, rx, ry, theta): rotation in image coords (y down)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)
    x = cx + rx * np.cos(t) * c - ry * np.sin(t) * s
    y = cy + rx * np.cos(t) * s + ry * np.sin(t) * c
    return list(zip(x.tolist(), y.tolist()))


def rasterise(regions, W, H):
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for g in regions:
        sa = g["shape_attributes"]
        n = sa["name"]
        if n == "polygon":
            pts = list(zip(sa["all_points_x"], sa["all_points_y"]))
            if len(pts) >= 3:
                d.polygon(pts, fill=1)
        elif n == "ellipse":
            d.polygon(_ellipse_poly(sa["cx"], sa["cy"], sa["rx"], sa["ry"], sa.get("theta", 0.0)), fill=1)
        elif n == "circle":
            d.polygon(_ellipse_poly(sa["cx"], sa["cy"], sa["r"], sa["r"]), fill=1)
        elif n == "rect":
            d.rectangle([sa["x"], sa["y"], sa["x"] + sa["width"], sa["y"] + sa["height"]], fill=1)
        else:
            raise ValueError(f"unsupported VIA shape: {n}")
    return np.array(m, dtype=bool)


def load_br35h_masks(ann_root, size=224):
    """Returns {md5: {'mask': bool[size,size], 'mask_native': bool[H,W], 'file': str,
    'split': str, 'n_regions': int, 'area_frac': float}} for every annotated image."""
    out = {}
    for jf in sorted(glob.glob(os.path.join(ann_root, "*", "annotations_*.json"))):
        split = os.path.basename(os.path.dirname(jf))
        with open(jf) as f:
            meta = json.load(f)
        meta = meta.get("_via_img_metadata", meta)
        for rec in meta.values():
            p = os.path.join(os.path.dirname(jf), rec["filename"])
            if not os.path.exists(p):
                continue
            regs = rec["regions"]
            regs = list(regs.values()) if isinstance(regs, dict) else regs
            W, H = Image.open(p).size
            nat = rasterise(regs, W, H)
            m = np.array(Image.fromarray(nat.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 127
            out[_md5(p)] = dict(mask=m, mask_native=nat, file=rec["filename"], split=split,
                                n_regions=len(regs), area_frac=float(m.mean()))
    return out


def attach_masks(test_df, masks, path_col="path"):
    """Adds 'md5' and 'has_mask' columns; returns (test_df, list_of_masks_or_None aligned to rows)."""
    md5s = [_md5(p) for p in test_df[path_col]]
    test_df = test_df.copy()
    test_df["md5"] = md5s
    test_df["has_mask"] = [h in masks for h in md5s]
    aligned = [masks[h]["mask"] if h in masks else None for h in md5s]
    return test_df, aligned


# ---------------------------------------------------------------- scan clusters
def scan_clusters(paths, phash_thr=5, mirror_corr=0.95):
    """Groups images that are the same scan: identical MD5, pHash Hamming <= phash_thr,
    or a left-right mirrored copy (pHash of the mirror <= phash_thr AND pixel correlation
    of the mirror > mirror_corr AND higher than the unmirrored correlation + 0.05).
    The pixel check is needed because axial slices are roughly symmetric, so mirrored
    pHashes alone collide for unrelated scans.
    On the full Br35H yes/no set (3,000 images) this gives 624 clusters
    (385 tumor, 239 no-tumor, 0 mixed-class); 801 annotated images -> 247 clusters.
    Returns np.array of cluster ids aligned with `paths`."""
    import imagehash
    from PIL import ImageOps
    n = len(paths)
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[rb] = ra

    gray = [Image.open(p).convert("L") for p in paths]
    H = np.array([np.asarray(imagehash.phash(g).hash).ravel() for g in gray], np.uint8)
    Hm = np.array([np.asarray(imagehash.phash(ImageOps.mirror(g)).hash).ravel() for g in gray], np.uint8)
    md5s = {}
    for i, p in enumerate(paths):
        h = _md5(p)
        if h in md5s:
            union(md5s[h], i)
        else:
            md5s[h] = i
    small = None

    def z(i, mirror=False):
        nonlocal small
        if small is None:
            small = {}
        k = (i, mirror)
        if k not in small:
            g = ImageOps.mirror(gray[i]) if mirror else gray[i]
            a = np.asarray(g.resize((128, 128)), np.float64)
            small[k] = (a - a.mean()) / (a.std() + 1e-8)
        return small[k]

    for i in range(n):
        d = (H[i + 1:] != H[i]).sum(1)
        for j in np.where(d <= phash_thr)[0] + i + 1:
            union(i, j)
        dm = (Hm[i + 1:] != H[i]).sum(1)
        for j in np.where(dm <= phash_thr)[0] + i + 1:
            if find(i) == find(j):
                continue
            r_m = float((z(i, True) * z(j)).mean())
            r_d = float((z(i) * z(j)).mean())
            if r_m > mirror_corr and r_m > r_d + 0.05:
                union(i, j)
    return np.array([find(i) for i in range(n)])


def leakage_flags(test_paths, train_paths):
    """For each test image: does any training (or validation) image belong to the same scan
    cluster? Returns (cluster_ids_for_test, bool array)."""
    allp = list(train_paths) + list(test_paths)
    cl = scan_clusters(allp)
    tr = set(cl[: len(train_paths)])
    te = cl[len(train_paths):]
    return te, np.array([c in tr for c in te])


# ---------------------------------------------------------------- brain mask
def brain_mask(img_u8):
    """Same definition as the notebook's _brain_mask (Otsu, largest component, dilated)."""
    import cv2
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY) if img_u8.ndim == 3 else img_u8
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n_cc <= 1:
        return np.ones_like(gray, dtype=bool)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = (lab == largest).astype(np.uint8)
    return cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=2).astype(bool)