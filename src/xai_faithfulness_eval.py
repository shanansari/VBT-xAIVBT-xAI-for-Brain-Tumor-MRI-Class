"""
Final XAI faithfulness and localisation evaluation for GCAF-CEViT (Br35H).
Responds to the reviewer comment on Section III.D / Table 8 / Figure 8.

For every attribution method, and for two random-attribution baselines:
  * Insertion AUC (higher = better) and Deletion AUC (lower = better), Petsiuk et al. 2018 (RISE)
  * Gap = Insertion - Deletion, and the gap minus the in-brain random gap
  * Pointing game against the expert Br35H tumor polygons/ellipses (not a brain mask),
    with 15-px tolerance (Zhang et al. 2016), the chance level for a random in-brain
    point, and energy-in-mask with its chance level.
Statistics treat copies of the same scan as ONE unit: per-image scores are averaged within
scan clusters, then Wilcoxon signed-rank (Holm-corrected) and cluster bootstrap CIs are
computed over clusters. (Br35H contains heavy duplication: 3,000 images -> 624 scan clusters.)

Strata: All / Tumor / No tumor / Correctly classified / Test images with no copy in training.

Usage inside the notebook: see notebook_xai_final_cells.py (cells A-E).
Self-test without data:  python xai_faithfulness_eval.py
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon, rankdata
from scipy.ndimage import binary_dilation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CFG = dict(
    img_size=224,
    n_steps=50,               # 2% of pixels per step; 100 is finer but 2x slower
    baseline="blur",          # primary: "blur"; sensitivity: "black" or "brain_mean"
    blur_kernel=31,
    blur_sigma=10.0,
    n_random_seeds=3,
    pointing_tolerance=15,
    target="pred",            # explain the PREDICTED class (notebook cell 51 rule)
    n_boot=2000,
    seed=0,
    batch=51,
)
TUMOR, NO_TUMOR = 1, 0
RANDOM_U, RANDOM_B = "Random (uniform)", "Random (in-brain)"


# ------------------------------------------------------------------ helpers
def gaussian_blur(x, k=31, sigma=10.0):
    ax = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)
    return x


def black_like(x):
    """Pixel value 0 (black) expressed in ImageNet-normalised space.
    torch.zeros_like(x) would be ImageNet-mean GREY, not black."""
    m = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return ((0 - m) / s).expand_as(x).clone()


def make_baseline(x, brain=None, cfg=CFG):
    b = cfg["baseline"]
    if b == "blur":
        return gaussian_blur(x, cfg["blur_kernel"], cfg["blur_sigma"])
    if b == "black":
        return black_like(x)
    if b == "brain_mean":
        bm = torch.as_tensor(brain, dtype=x.dtype, device=x.device)[None, None]
        mu = (x * bm).sum((2, 3), keepdim=True) / bm.sum().clamp(min=1)
        return x * (1 - bm) + mu * bm
    raise ValueError(b)


@torch.no_grad()
def probs(forward, x):
    return F.softmax(forward(x), dim=1)


@torch.no_grad()
def causal_curves(forward, x, sal, target, base, cfg=CFG):
    """Returns (insertion_curve, deletion_curve), each (n_steps+1,)."""
    H, W = sal.shape
    n = H * W
    order = torch.as_tensor(np.argsort(-sal.reshape(-1), kind="stable").copy(), device=x.device)
    steps = cfg["n_steps"]
    counts = [int(round(i * n / steps)) for i in range(steps + 1)]
    rank = torch.empty(n, device=x.device, dtype=torch.long)
    rank[order] = torch.arange(n, device=x.device)
    out = {}
    for mode in ("insertion", "deletion"):
        start, finish = (base, x) if mode == "insertion" else (x, base)
        vals = []
        for j in range(0, len(counts), cfg["batch"]):
            cs = torch.tensor(counts[j:j + cfg["batch"]], device=x.device)
            m = (rank[None, :] < cs[:, None]).float().view(-1, 1, H, W)
            imgs = start * (1 - m) + finish * m
            vals.append(probs(forward, imgs)[:, target].cpu())
        out[mode] = torch.cat(vals).numpy()
    return out["insertion"], out["deletion"]


def curve_auc(c):
    xs = np.linspace(0, 1, len(c))
    f = getattr(np, "trapezoid", None) or np.trapz
    return float(f(c, xs))


def to_hw(sal, size):
    sal = np.asarray(sal, dtype=np.float64)
    if sal.ndim == 3:
        sal = sal.sum(0)
    if sal.shape != (size, size):
        t = torch.as_tensor(sal)[None, None].float()
        sal = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    return np.nan_to_num(sal)


def random_map(shape, rng, brain=None):
    r = rng.random(shape)
    if brain is not None:
        r = np.where(brain, r, r - 2.0)   # all brain pixels ranked before background
    return r


def pointing(sal, mask, brain, tol):
    """Hit if the argmax is inside the tumor mask dilated by tol px.
    chance = P(uniform random in-brain point lands in the dilated mask)."""
    md = binary_dilation(mask, iterations=tol) if tol > 0 else mask
    y, x = np.unravel_index(np.argmax(sal), sal.shape)
    hit = bool(md[y, x])
    chance = float((md & brain).sum() / max(brain.sum(), 1))
    s = np.clip(sal, 0, None)
    energy = float(s[mask].sum() / (s.sum() + 1e-12))
    energy_chance = float(mask.sum() / max(brain.sum(), 1))
    return hit, chance, energy, energy_chance


# --------------------------------------------------------------- statistics
def cluster_means(values, clusters):
    v, c = np.asarray(values, float), np.asarray(clusters)
    u = np.unique(c)
    return np.array([v[c == k].mean() for k in u])


def cluster_boot_ci(values, clusters, n_boot, rng):
    v, c = np.asarray(values, float), np.asarray(clusters)
    if len(v) == 0:
        return (np.nan, np.nan)
    u = np.unique(c)
    groups = [v[c == k] for k in u]
    sums = np.array([g.sum() for g in groups]); lens = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(u), (n_boot, len(u)))
    m = sums[idx].sum(1) / lens[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (n - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def paired_cluster_test(a, b, clusters, alternative):
    """Wilcoxon on cluster-level means of (a - b). Returns (p, rank-biserial, n_clusters)."""
    d = cluster_means(np.asarray(a) - np.asarray(b), clusters)
    nz = d[np.abs(d) > 1e-12]
    if len(nz) < 6:
        return np.nan, np.nan, len(d)
    _, p = wilcoxon(nz, alternative=alternative)
    r = rankdata(np.abs(nz))
    return float(p), float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()), len(d)


# --------------------------------------------------------------------- main
def evaluate(forward, items, explainers, cfg=CFG, out_dir="xai_results", log_every=10):
    """
    forward   : callable x -> logits (B, 2)   (use the notebook's `wrapped`)
    items     : list of dicts with keys
                x (1,3,S,S) normalised tensor on the model device, label, pred, prob (p_tumor),
                id, cluster (scan-cluster id), leak (bool: same scan in train/val),
                brain (bool S x S), mask (bool S x S tumor mask or None)
    explainers: {name: fn(x, target) -> 2-D (or C x H x W) array}
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    S = cfg["img_size"]
    names = list(explainers) + [RANDOM_U, RANDOM_B]
    rows = []
    for k, it in enumerate(items):
        x, y, pred = it["x"], int(it["label"]), int(it["pred"])
        t = pred if cfg["target"] == "pred" else y
        brain = it["brain"]
        base = make_baseline(x, brain, cfg)
        maps = {}
        for n, fn in explainers.items():
            try:
                maps[n] = to_hw(fn(x, t), S)
            except Exception as e:  # keep going, but record the failure
                print(f"[{it['id']}] {n} failed: {e}")
                maps[n] = None
        # tumor-class maps for the pointing game when the prediction is not 'tumor' (FN)
        pg_maps = maps
        if y == TUMOR and it.get("mask") is not None and t != TUMOR:
            pg_maps = {}
            for n, fn in explainers.items():
                try:
                    pg_maps[n] = to_hw(fn(x, TUMOR), S)
                except Exception:
                    pg_maps[n] = None
        for n in names:
            r = dict(id=it["id"], method=n, label=y, pred=pred, target=t, prob_tumor=float(it["prob"]),
                     cluster=str(it["cluster"]), leak=bool(it["leak"]), has_mask=it.get("mask") is not None)
            if n in (RANDOM_U, RANDOM_B):
                ins, dele, hits, en = [], [], [], []
                for _ in range(cfg["n_random_seeds"]):
                    sal = random_map((S, S), rng, brain if n == RANDOM_B else None)
                    ci, cd = causal_curves(forward, x, sal, t, base, cfg)
                    ins.append(curve_auc(ci)); dele.append(curve_auc(cd))
                    if y == TUMOR and r["has_mask"]:
                        h, ch, e, ech = pointing(sal, it["mask"], brain, cfg["pointing_tolerance"])
                        hits.append(h); en.append(e)
                r.update(ins=float(np.mean(ins)), dele=float(np.mean(dele)))
                if hits:
                    r.update(hit=float(np.mean(hits)), chance=ch, energy=float(np.mean(en)), energy_chance=ech)
            else:
                sal = maps[n]
                if sal is None:
                    continue
                ci, cd = causal_curves(forward, x, sal, t, base, cfg)
                r.update(ins=curve_auc(ci), dele=curve_auc(cd))
                if y == TUMOR and r["has_mask"] and pg_maps.get(n) is not None:
                    h, ch, e, ech = pointing(pg_maps[n], it["mask"], brain, cfg["pointing_tolerance"])
                    r.update(hit=float(h), chance=ch, energy=e, energy_chance=ech)
            r["gap"] = r["ins"] - r["dele"]
            rows.append(r)
        if log_every and (k + 1) % log_every == 0:
            print(f"  {k + 1}/{len(items)} images done")
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "xai_per_image.csv"), index=False)
    summary = summarise(df, list(explainers), cfg, rng)
    with open(os.path.join(out_dir, "xai_summary.json"), "w") as f:
        json.dump({"config": cfg, "summary": summary}, f, indent=2, default=float)
    with open(os.path.join(out_dir, "table8.tex"), "w") as f:
        f.write(latex_table8(summary))
    with open(os.path.join(out_dir, "table8.md"), "w") as f:
        f.write(markdown_table8(summary))
    return df, summary


def summarise(df, methods, cfg, rng):
    strata = {
        "All": df.index == df.index,
        "Tumor": df.label == TUMOR,
        "No tumor": df.label == NO_TUMOR,
        "Correct": df.label == df.pred,
        "No train copy": ~df.leak,
    }
    out = {}
    names = methods + [RANDOM_U, RANDOM_B]
    for s, sel in strata.items():
        out[s] = {}
        sub = df[sel]
        refB = sub[sub.method == RANDOM_B].set_index("id")
        refU = sub[sub.method == RANDOM_U].set_index("id")
        raw, keys = [], []
        for n in names:
            d = sub[sub.method == n].set_index("id")
            if len(d) == 0:
                continue
            e = dict(n_images=int(len(d)), n_clusters=int(d.cluster.nunique()))
            for col in ("ins", "dele", "gap"):
                cm = cluster_means(d[col], d.cluster)
                e[f"{col}_mean"] = float(d[col].mean())
                e[f"{col}_sd_clusters"] = float(cm.std(ddof=1)) if len(cm) > 1 else float("nan")
                e[f"{col}_ci"] = cluster_boot_ci(d[col], d.cluster, cfg["n_boot"], rng)
            ids = d.index.intersection(refB.index)
            e["gap_minus_random_brain"] = float((d.loc[ids, "gap"] - refB.loc[ids, "gap"]).mean()) if len(ids) else float("nan")
            if n in methods:
                for ref, rname in ((refB, "brain"), (refU, "uniform")):
                    ids = d.index.intersection(ref.index)
                    cl = d.loc[ids, "cluster"]
                    p_i, r_i, nc = paired_cluster_test(d.loc[ids, "ins"], ref.loc[ids, "ins"], cl, "greater")
                    p_d, r_d, _ = paired_cluster_test(d.loc[ids, "dele"], ref.loc[ids, "dele"], cl, "less")
                    e[f"ins_rbc_vs_{rname}"], e[f"del_rbc_vs_{rname}"] = r_i, r_d
                    raw += [p_i, p_d]; keys += [(n, f"ins_p_holm_vs_{rname}"), (n, f"del_p_holm_vs_{rname}")]
            out[s][n] = e
        valid = [i for i, p in enumerate(raw) if not np.isnan(p)]
        adj = holm([raw[i] for i in valid]) if valid else []
        for k, (n, key) in enumerate(keys):
            out[s][n][key] = float(adj[valid.index(k)]) if k in valid else None
    # pointing game on tumor images with expert masks
    pg = {}
    pgdf = df[(df.label == TUMOR) & df.has_mask & df.hit.notna()] if "hit" in df else df.iloc[0:0]
    raw, keys = [], []
    for n in names:
        d = pgdf[pgdf.method == n]
        if len(d) == 0:
            pg[n] = dict(n_images=0)
            continue
        e = dict(n_images=int(len(d)), n_clusters=int(d.cluster.nunique()),
                 hit_rate=float(d.hit.mean()), hit_ci=cluster_boot_ci(d.hit, d.cluster, cfg["n_boot"], rng),
                 chance=float(d.chance.mean()), energy=float(d.energy.mean()),
                 energy_chance=float(d.energy_chance.mean()),
                 energy_ci=cluster_boot_ci(d.energy, d.cluster, cfg["n_boot"], rng))
        if n in methods:
            p, r, _ = paired_cluster_test(d.hit, d.chance, d.cluster, "greater")
            pe, re_, _ = paired_cluster_test(d.energy, d.energy_chance, d.cluster, "greater")
            e.update(hit_rbc=r, energy_rbc=re_)
            raw += [p, pe]; keys += [(n, "hit_p_holm_vs_chance"), (n, "energy_p_holm_vs_chance")]
        pg[n] = e
    valid = [i for i, p in enumerate(raw) if not np.isnan(p)]
    adj = holm([raw[i] for i in valid]) if valid else []
    for k, (n, key) in enumerate(keys):
        pg[n][key] = float(adj[valid.index(k)]) if k in valid else None
    out["Pointing game (tumor images with expert mask)"] = pg
    return out


# ------------------------------------------------------------------ tables
def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def markdown_table8(s):
    T, N, A = s["Tumor"], s["No tumor"], s["All"]
    pg = s["Pointing game (tumor images with expert mask)"]
    L = ["| Method | Ins (tumor) | Del (tumor) | Gap (tumor) | Ins (no tumor) | Del (no tumor) | "
         "Gap - random (all) | p ins / del vs in-brain random (all) | Pointing hit (n) | Chance | Energy in mask (chance) |",
         "|" + "---|" * 11]
    for n in A:
        t, nt, a, p = T.get(n, {}), N.get(n, {}), A[n], pg.get(n, {})
        f = lambda e, k: f"{e[k]:.3f}" if k in e else "--"
        if not p.get("n_images"):
            hit = "--"
        elif n in (RANDOM_U, RANDOM_B):
            hit = f"{100 * p['hit_rate']:.1f}% (n={p['n_images']})"
        else:
            hit = f"{100 * p['hit_rate']:.1f}% ({int(round(p['hit_rate'] * p['n_images']))}/{p['n_images']})"
        ch = f"{100 * p['chance']:.1f}%" if p.get("n_images") else "--"
        en = f"{p['energy']:.3f} ({p['energy_chance']:.3f})" if p.get("n_images") else "--"
        pv = f"{_fmt_p(a.get('ins_p_holm_vs_brain'))} / {_fmt_p(a.get('del_p_holm_vs_brain'))}" if "ins_p_holm_vs_brain" in a else "ref."
        L.append(f"| {n} | {f(t, 'ins_mean')} | {f(t, 'dele_mean')} | {f(t, 'gap_mean')} | {f(nt, 'ins_mean')} | "
                 f"{f(nt, 'dele_mean')} | {a['gap_minus_random_brain']:+.3f} | {pv} | {hit} | {ch} | {en} |")
    nA = next(iter(A.values()))
    L.append(f"\nImages: {nA['n_images']} ({nA['n_clusters']} scan clusters). Pointing game: tumor images with expert masks. "
             "Stats are over scan clusters (Wilcoxon, Holm).")
    return "\n".join(L) + "\n"


def latex_table8(s):
    T, N, A = s["Tumor"], s["No tumor"], s["All"]
    pg = s["Pointing game (tumor images with expert mask)"]
    out = [r"\begin{table*}[t]\centering",
           r"\caption{Per-method causal faithfulness (insertion/deletion AUC, blur baseline) and tumor localisation "
           r"(pointing game against the Br35H expert annotations, 15-px tolerance). Statistics are computed over scan "
           r"clusters; $p$-values: one-sided Wilcoxon vs.\ in-brain random attribution, Holm-corrected.}",
           r"\label{tab:xai}",
           r"\begin{tabular}{lcccccccc}\hline",
           r"Method & Ins$\uparrow$ (T) & Del$\downarrow$ (T) & Ins$\uparrow$ (NT) & Del$\downarrow$ (NT) & "
           r"Gap$-$rand & $p_{ins}/p_{del}$ & Pointing (n) & Chance \\ \hline"]
    for n in A:
        t, nt, a, p = T.get(n, {}), N.get(n, {}), A[n], pg.get(n, {})
        f = lambda e, k: f"{e[k]:.3f}" if k in e else "--"
        hit = f"{100 * p['hit_rate']:.1f}\\% ({p['n_images']})" if p.get("n_images") else "--"
        ch = f"{100 * p['chance']:.1f}\\%" if p.get("n_images") else "--"
        pv = f"{_fmt_p(a.get('ins_p_holm_vs_brain'))}/{_fmt_p(a.get('del_p_holm_vs_brain'))}" if "ins_p_holm_vs_brain" in a else "ref."
        out.append(f"{n} & {f(t, 'ins_mean')} & {f(t, 'dele_mean')} & {f(nt, 'ins_mean')} & {f(nt, 'dele_mean')} & "
                   f"{a['gap_minus_random_brain']:+.3f} & {pv.replace('<', '$<$')} & {hit} & {ch} \\\\")
    out += [r"\hline\end{tabular}\end{table*}"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------- figure 8
def figure8(items, explainers, case_idx, out_path, class_names=("No tumor", "Tumor"), cfg=CFG):
    """case_idx: {'TP': [...], 'FN': [...], 'TN': [...], 'FP': [...]} indices into items.
    Rows = cases, columns = original + each method; expert tumor mask outlined in green."""
    import matplotlib
    import matplotlib.pyplot as plt
    order = [(k, i) for k in ("TP", "FN", "TN", "FP") for i in case_idx.get(k, [])]
    m = list(explainers)
    fig, axes = plt.subplots(len(order), len(m) + 1, figsize=(2.3 * (len(m) + 1), 2.9 * len(order)), squeeze=False)
    mean = np.array(IMAGENET_MEAN)[:, None, None]; std = np.array(IMAGENET_STD)[:, None, None]
    for r, (kind, i) in enumerate(order):
        it = items[i]
        img = np.clip(it["x"][0].detach().cpu().numpy() * std + mean, 0, 1).transpose(1, 2, 0)
        t = int(it["pred"]) if cfg["target"] == "pred" else int(it["label"])
        conf = it["prob"] if t == 1 else 1 - it["prob"]
        ax = axes[r, 0]; ax.imshow(img)
        if it.get("mask") is not None:
            ax.contour(it["mask"], levels=[0.5], colors="lime", linewidths=1.0)
        ax.set_title(f"{kind}: true {class_names[it['label']]}\npred {class_names[it['pred']]} ({conf:.2f})", fontsize=7)
        for c, n in enumerate(m, start=1):
            ax = axes[r, c]
            try:
                sal = to_hw(explainers[n](it["x"], t), cfg["img_size"])
                # display only: positive evidence for the target class, clipped at the 99th percentile
                # (signed IG/SHAP maps would otherwise put zero at mid-colour and wash out the panel)
                sal = np.clip(sal, 0, None)
                hi = np.percentile(sal, 99) if sal.max() > 0 else 1.0
                sal = np.clip(sal / (hi + 1e-12), 0, 1)
                if n in ("Integrated Gradients", "SHAP (GradientSHAP)"):
                    from scipy.ndimage import gaussian_filter
                    sal = gaussian_filter(sal, 2); sal = sal / (sal.max() + 1e-12)
                ax.imshow(img); ax.imshow(sal, cmap="jet", alpha=0.45)
                if it.get("mask") is not None:
                    ax.contour(it["mask"], levels=[0.5], colors="lime", linewidths=1.0)
            except Exception as e:
                ax.text(0.5, 0.5, "failed", ha="center")
            if r == 0:
                ax.set_title(n, fontsize=8)
        for ax in axes[r]:
            ax.axis("off")
    plt.tight_layout(h_pad=1.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- adapters
def build_explainers(model, forward, cfg=CFG, lime_samples=500, scorecam_batch=64, which=None):
    """Adapters for the notebook's HybridModel (cell 16):
         model.cnn.cbam        -> CBAM-refined (B,1280,7,7) map  (Grad-CAM++, Score-CAM)
         model.vit.vit.blocks  -> timm ViT blocks                (attention rollout, ViT branch only)
       `forward` must return logits only (the notebook's `wrapped`).
       Attribution sign convention: positive = evidence FOR the target class."""
    from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from captum.attr import IntegratedGradients, Occlusion, Lime, GradientShap
    from captum._utils.models.linear_model import SkLearnLinearRegression

    layer = [model.cnn.cbam]
    ex = {}

    def f_gcpp(x, t):
        with GradCAMPlusPlus(model=forward, target_layers=layer) as cam:
            return cam(input_tensor=x, targets=[ClassifierOutputTarget(t)])[0]
    ex["Grad-CAM++"] = f_gcpp

    def f_scam(x, t):
        with ScoreCAM(model=forward, target_layers=layer) as cam:
            cam.batch_size = scorecam_batch
            return cam(input_tensor=x, targets=[ClassifierOutputTarget(t)])[0]
    ex["Score-CAM"] = f_scam

    def f_rollout(x, t):
        blocks = model.vit.vit.blocks
        att, hooks = [], []
        for b in blocks:
            b.attn.fused_attn = False
            hooks.append(b.attn.attn_drop.register_forward_hook(lambda _m, _i, o: att.append(o.detach())))
        try:
            with torch.no_grad():
                forward(x)
        finally:
            for h in hooks:
                h.remove()
        R = torch.eye(att[0].shape[-1], device=att[0].device)
        for A in att:
            A = A[0].mean(0)
            A = A + torch.eye(A.shape[-1], device=A.device)
            A = A / A.sum(-1, keepdim=True)
            R = A @ R
        g = int((R.shape[-1] - 1) ** 0.5)
        return R[0, 1:].reshape(g, g).cpu().numpy()   # class-agnostic: same map for both targets
    ex["Attention rollout (ViT)"] = f_rollout

    ig = IntegratedGradients(forward)
    def f_ig(x, t):
        a = ig.attribute(x, baselines=gaussian_blur(x, cfg["blur_kernel"], cfg["blur_sigma"]),
                         target=t, n_steps=32, internal_batch_size=16)
        return a[0].sum(0).detach().cpu().numpy()
    ex["Integrated Gradients"] = f_ig

    gs = GradientShap(forward)
    def f_shap(x, t):
        base = torch.cat([gaussian_blur(x, cfg["blur_kernel"], cfg["blur_sigma"]), black_like(x)])
        a = gs.attribute(x, baselines=base, target=t, n_samples=32, stdevs=0.05)
        return a[0].sum(0).detach().cpu().numpy()
    ex["SHAP (GradientSHAP)"] = f_shap

    occ = Occlusion(forward)
    def f_occ(x, t):
        a = occ.attribute(x, target=t, sliding_window_shapes=(3, 32, 32), strides=(3, 16, 16),
                          baselines=gaussian_blur(x, cfg["blur_kernel"], cfg["blur_sigma"]), perturbations_per_eval=32)
        return a[0].sum(0).detach().cpu().numpy()
    ex["Occlusion"] = f_occ

    lime = Lime(forward, interpretable_model=SkLearnLinearRegression())
    def f_lime(x, t):
        from skimage.segmentation import slic
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1); std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = (x[0].detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        seg = slic(img, n_segments=50, compactness=10, start_label=0)
        fm = torch.as_tensor(seg, device=x.device)[None, None].expand(1, 3, -1, -1)
        a = lime.attribute(x, target=t, feature_mask=fm, n_samples=lime_samples, perturbations_per_eval=32,
                           baselines=gaussian_blur(x, cfg["blur_kernel"], cfg["blur_sigma"]))
        return a[0, 0].detach().cpu().numpy()
    ex["LIME"] = f_lime

    if which:
        ex = {k: v for k, v in ex.items() if k in which}
    return ex


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    torch.manual_seed(0)
    S = 224

    class Toy(torch.nn.Module):
        """Tumor logit is driven ONLY by the brightness inside a lesion region."""
        def forward(self, x):
            z = 12 * (x[:, :, 95:125, 95:125].mean((1, 2, 3)) - 1.0)
            return torch.stack([-z, z], 1)

    toy = Toy()
    items = []
    for i in range(16):
        x = black_like(torch.zeros(1, 3, S, S))
        x[:, :, 30:194, 30:194] = 0.0
        lab = i % 2
        if lab:
            x[:, :, 95:125, 95:125] = 2.0
        brain = np.zeros((S, S), bool); brain[30:194, 30:194] = True
        mask = np.zeros((S, S), bool); mask[95:125, 95:125] = True
        p = float(F.softmax(toy(x), 1)[0, 1])
        items.append(dict(x=x, label=lab, pred=int(p > 0.5), prob=p, id=str(i), cluster=str(i // 2),
                          leak=False, brain=brain, mask=mask if lab else None))

    def oracle(x, t):
        s = np.zeros((S, S)); s[95:125, 95:125] = 1.0
        return s + 1e-3 * np.random.rand(S, S)

    def anti(x, t):
        s = np.ones((S, S)); s[95:125, 95:125] = 0.0
        return s + 1e-3 * np.random.rand(S, S)

    cfg = dict(CFG, n_steps=20, n_random_seeds=2, n_boot=300, baseline="black")
    df, s = evaluate(toy, items, {"Oracle": oracle, "Anti-oracle": anti}, cfg=cfg, out_dir="/tmp/xai_selftest", log_every=0)
    T = s["Tumor"]; pg = s["Pointing game (tumor images with expert mask)"]
    print(open("/tmp/xai_selftest/table8.md").read())
    assert T["Oracle"]["ins_mean"] > T[RANDOM_B]["ins_mean"] and T["Oracle"]["dele_mean"] < T[RANDOM_B]["dele_mean"]
    assert T["Anti-oracle"]["dele_mean"] > T["Oracle"]["dele_mean"]
    assert T["Oracle"]["gap_mean"] > 0.3 and T["Anti-oracle"]["gap_mean"] < 0
    assert pg["Oracle"]["hit_rate"] == 1.0 and pg["Anti-oracle"]["hit_rate"] == 0.0
    assert abs(pg[RANDOM_B]["hit_rate"] - pg[RANDOM_B]["chance"]) < 0.25
    print("self-test passed")