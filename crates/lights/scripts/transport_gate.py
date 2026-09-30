import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

KEY_CENTRE = (-0.35, -0.50)
KEY_HALF = 0.12
KEY_SHARE = 0.5
JND = 2.3

THEMES = {
    "dark": {"ground": (0.012, 0.012, 0.016), "overlay": (0.05, 0.05, 0.06)},
    "light": {"ground": (0.80, 0.80, 0.82), "overlay": (0.90, 0.90, 0.90)},
}

CARD_INKS = [
    (0.80, 0.12, 0.08),
    (0.12, 0.62, 0.16),
    (0.10, 0.22, 0.85),
    (0.90, 0.50, 0.06),
]


def corner(x, y):
    a = np.sqrt(1.0 + x * x)
    b = np.sqrt(1.0 + y * y)
    return (x / a * np.arctan(y / a) + y / b * np.arctan(x / b)) / (2.0 * math.pi)


def rect_factor(u0, u1, v0, v1):
    return corner(u1, v1) - corner(u0, v1) - corner(u1, v0) + corner(u0, v0)


KEY_U0, KEY_U1 = KEY_CENTRE[0] - KEY_HALF, KEY_CENTRE[0] + KEY_HALF
KEY_V0, KEY_V1 = KEY_CENTRE[1] - KEY_HALF, KEY_CENTRE[1] + KEY_HALF
KEY_FACTOR = float(rect_factor(KEY_U0, KEY_U1, KEY_V0, KEY_V1))
L_SKY = 1.0 - KEY_SHARE
L_KEY = KEY_SHARE / KEY_FACTOR


def scene(theme, scale):
    s = float(scale)
    ink = THEMES[theme]
    cards = []
    w, h, gap, margin = 120.0, 72.0, 28.0, 36.0
    heights = [2.0, 4.0, 8.0, 16.0]
    for i, (dz, rho) in enumerate(zip(heights, CARD_INKS)):
        cx, cy = i % 2, i // 2
        x0 = margin + cx * (w + gap)
        y0 = margin + cy * (h + gap)
        cards.append((x0 * s, (x0 + w) * s, y0 * s, (y0 + h) * s, dz * s, rho))
    ox0, oy0 = margin + w - 30.0, margin + h - 18.0
    cards.append((ox0 * s, (ox0 + 84.0) * s, oy0 * s, (oy0 + 40.0) * s, 24.0 * s, ink["overlay"]))
    width = int(round((2 * margin + 2 * w + gap) * s))
    height = int(round((2 * margin + 2 * h + gap) * s))
    return {"cards": cards, "ground": ink["ground"], "width": width, "height": height}


def arrays(sc):
    c = sc["cards"]
    return (np.array([k[0] for k in c]), np.array([k[1] for k in c]), np.array([k[2] for k in c]),
            np.array([k[3] for k in c]), np.array([k[4] for k in c]), np.array([k[5] for k in c]))


def primary(sc, xs, ys):
    x0, x1, y0, y1, hz, rho = arrays(sc)
    z = np.zeros_like(xs)
    alb = np.tile(np.array(sc["ground"]), (xs.size, 1))
    for i in np.argsort(hz):
        inside = (xs >= x0[i]) & (xs < x1[i]) & (ys >= y0[i]) & (ys < y1[i])
        z[inside] = hz[i]
        alb[inside] = rho[i]
    return z, alb


def nearest_up(sc, px, py, pz, dx, dy, dz):
    x0, x1, y0, y1, hz, rho = arrays(sc)
    best = np.full(px.shape, np.inf)
    idx = np.full(px.shape, -1)
    for i in range(hz.size):
        t = (hz[i] - pz) / dz
        ok = (hz[i] > pz + 1e-9) & (t < best)
        hx = px + t * dx
        hy = py + t * dy
        ok &= (hx >= x0[i]) & (hx < x1[i]) & (hy >= y0[i]) & (hy < y1[i])
        best = np.where(ok, t, best)
        idx = np.where(ok, i, idx)
    return idx, best


def nearest_down(sc, px, py, pz, dx, dy, dz):
    x0, x1, y0, y1, hz, rho = arrays(sc)
    best = -pz / dz
    idx = np.full(px.shape, -1)
    for i in range(hz.size):
        t = (hz[i] - pz) / dz
        ok = (hz[i] < pz - 1e-9) & (t < best)
        hx = px + t * dx
        hy = py + t * dy
        ok &= (hx >= x0[i]) & (hx < x1[i]) & (hy >= y0[i]) & (hy < y1[i])
        best = np.where(ok, t, best)
        idx = np.where(ok, i, idx)
    return idx, best


def cosine(u1, u2, up):
    r = np.sqrt(u1)
    phi = 2.0 * math.pi * u2
    dz = np.sqrt(np.maximum(1.0 - u1, 1e-12))
    return r * np.cos(phi), r * np.sin(phi), np.where(up, dz, -dz)


def key_visible(sc, px, py, pz, u, v):
    norm = np.sqrt(1.0 + u * u + v * v)
    idx, _ = nearest_up(sc, px, py, pz, u / norm, v / norm, 1.0 / norm)
    return idx < 0


def key_weight(u, v):
    return 1.0 / (math.pi * (1.0 + u * u + v * v) ** 2)


def strata(n, rng):
    side = int(round(math.sqrt(n)))
    g = (np.arange(side) + 0.5) / side
    a, b = np.meshgrid(g, g)
    base = np.stack([a.ravel(), b.ravel()], 1)
    jitter = (rng.random(base.shape) - 0.5) / side
    return base + jitter


def trace_rows(args):
    theme, scale, row0, row1, spp, kspp, depth, seed = args
    sc = scene(theme, scale)
    hz_c = arrays(sc)[4]
    rho_c = arrays(sc)[5]
    ground = np.array(sc["ground"])
    rng = np.random.default_rng(seed + row0)
    W = sc["width"]
    ys, xs = np.mgrid[row0:row1, 0:W]
    xs = xs.ravel() + 0.5
    ys = ys.ravel() + 0.5
    P = xs.size
    pz, alb = primary(sc, xs, ys)
    area = (KEY_U1 - KEY_U0) * (KEY_V1 - KEY_V0)
    ks = strata(kspp, rng)
    key = np.zeros(P)
    rot = rng.random((P, 2))
    for j in range(ks.shape[0]):
        uv = (ks[j] + rot) % 1.0
        u = KEY_U0 + uv[:, 0] * (KEY_U1 - KEY_U0)
        v = KEY_V0 + uv[:, 1] * (KEY_V1 - KEY_V0)
        key += (~key_visible(sc, xs, ys, pz, u, v)) * key_weight(u, v)
    key = KEY_SHARE - L_KEY * area * key / ks.shape[0]
    sky = np.zeros(P)
    bounce = np.zeros((P, 3))
    ss = strata(spp, rng)
    rot = rng.random((P, 2))
    for j in range(ss.shape[0]):
        uv = (ss[j] + rot) % 1.0
        dx, dy, dz = cosine(uv[:, 0], uv[:, 1], np.ones(P, bool))
        idx, t = nearest_up(sc, xs, ys, pz, dx, dy, dz)
        sky += (idx < 0) * L_SKY
        owner = np.nonzero(idx >= 0)[0]
        if owner.size == 0:
            continue
        thr = rho_c[idx[owner]]
        qx = xs[owner] + t[owner] * dx[owner]
        qy = ys[owner] + t[owner] * dy[owner]
        qz = hz_c[idx[owner]]
        up = False
        for _ in range(depth):
            r = rng.random((qx.size, 4))
            dx2, dy2, dz2 = cosine(r[:, 0], r[:, 1], np.full(qx.size, up))
            if up:
                u = KEY_U0 + r[:, 2] * (KEY_U1 - KEY_U0)
                v = KEY_V0 + r[:, 3] * (KEY_V1 - KEY_V0)
                gain = KEY_SHARE - L_KEY * area * (~key_visible(sc, qx, qy, qz, u, v)) * key_weight(u, v)
                idx2, t2 = nearest_up(sc, qx, qy, qz, dx2, dy2, dz2)
                gain = gain + (idx2 < 0) * L_SKY
                np.add.at(bounce, owner, thr * gain[:, None])
                keep = idx2 >= 0
                nxt = rho_c[np.maximum(idx2, 0)]
                nz = hz_c[np.maximum(idx2, 0)]
            else:
                idx2, t2 = nearest_down(sc, qx, qy, qz, dx2, dy2, dz2)
                keep = np.ones(qx.size, bool)
                hit = (idx2 >= 0)[:, None]
                nxt = np.where(hit, rho_c[np.maximum(idx2, 0)], ground)
                nz = np.where(idx2 >= 0, hz_c[np.maximum(idx2, 0)], 0.0)
            sel = np.nonzero(keep)[0]
            thr = thr[sel] * nxt[sel]
            qx = (qx + t2 * dx2)[sel]
            qy = (qy + t2 * dy2)[sel]
            qz = nz[sel]
            owner = owner[sel]
            up = not up
            if owner.size == 0:
                break
    direct = alb * (sky / ss.shape[0] + key)[:, None]
    indirect = alb * bounce / ss.shape[0]
    return row0, direct.reshape(row1 - row0, W, 3), indirect.reshape(row1 - row0, W, 3)


def render_reference(theme, scale, spp, kspp, depth, seed, workers, band=8):
    sc = scene(theme, scale)
    H, W = sc["height"], sc["width"]
    jobs = [(theme, scale, r, min(r + band, H), spp, kspp, depth, seed) for r in range(0, H, band)]
    direct = np.zeros((H, W, 3))
    indirect = np.zeros((H, W, 3))
    with Pool(workers) as pool:
        for row0, d, i in pool.imap_unordered(trace_rows, jobs):
            direct[row0:row0 + d.shape[0]] = d
            indirect[row0:row0 + d.shape[0]] = i
    return sc, direct, indirect


def screen_space(sc, steps=24, azimuths=16):
    H, W = sc["height"], sc["width"]
    ys, xs = np.mgrid[0:H, 0:W]
    px = xs.ravel() + 0.5
    py = ys.ravel() + 0.5
    hp, alb = primary(sc, px, py)
    hbuf = hp.reshape(H, W)
    x0, x1, y0, y1, hz, _ = arrays(sc)
    reach = 12.0 * float(hz.max()) if hz.size else 1.0
    radii = np.geomspace(0.75, reach, steps)
    blocked = np.zeros(H * W)
    for a in range(azimuths):
        phi = 2.0 * math.pi * (a + 0.5) / azimuths
        horizon = np.zeros(H * W)
        for d in radii:
            qx = np.floor(px + d * math.cos(phi)).astype(int)
            qy = np.floor(py + d * math.sin(phi)).astype(int)
            inside = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
            hq = np.where(inside, hbuf[np.clip(qy, 0, H - 1), np.clip(qx, 0, W - 1)], 0.0)
            horizon = np.maximum(horizon, np.arctan2(hq - hp, d))
        blocked += np.sin(horizon) ** 2
    ao = 1.0 - blocked / azimuths
    vis = np.ones(H * W)
    key_area = (KEY_U1 - KEY_U0) * (KEY_V1 - KEY_V0)
    for i in range(hz.size):
        dz = hz[i] - hp
        above = dz > 1e-9
        safe = np.where(above, dz, 1.0)
        u0 = np.maximum((x0[i] - px) / safe, KEY_U0)
        u1 = np.minimum((x1[i] - px) / safe, KEY_U1)
        v0 = np.maximum((y0[i] - py) / safe, KEY_V0)
        v1 = np.minimum((y1[i] - py) / safe, KEY_V1)
        frac = np.clip(u1 - u0, 0, None) * np.clip(v1 - v0, 0, None) / key_area
        vis *= np.where(above, 1.0 - frac, 1.0)
    plain = alb * (L_SKY * ao + L_KEY * KEY_FACTOR * vis)[:, None]
    bounced = alb * (L_SKY * multibounce(ao, alb) + (L_KEY * KEY_FACTOR * vis)[:, None])
    return plain.reshape(H, W, 3), bounced.reshape(H, W, 3), (hp.reshape(H, W) > 0)


def srgb_encode(lin):
    c = np.clip(lin, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def lab(lin):
    c = np.clip(lin, 0.0, 1.0)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180, np.where(hs < 360, hs / 2 + 180, hs / 2 - 180), hs / 2)
    hbp = np.where(C1p * C2p == 0, hs, hbp)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def quantise(lin):
    return np.round(srgb_encode(lin) * 255.0) / 255.0


def linear_from_srgb(s):
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def delta_e(a, b):
    return ciede2000(lab(linear_from_srgb(quantise(a))), lab(linear_from_srgb(quantise(b))))


def stats(de, mask=None):
    v = de[mask] if mask is not None else de.ravel()
    if v.size == 0:
        return {"pixels": 0}
    return {"pixels": int(v.size), "max": round(float(v.max()), 3),
            "p99": round(float(np.percentile(v, 99)), 3), "p999": round(float(np.percentile(v, 99.9)), 3),
            "over_1": int((v > 1.0).sum()), "over_jnd": int((v > JND).sum())}


def save(path, lin):
    Image.fromarray((quantise(lin) * 255).astype(np.uint8)).save(path)


def heat(de):
    t = np.clip(de / (2 * JND), 0, 1)
    rgb = np.stack([t, np.clip(2 * t - 1, 0, 1) * 0.3, 1 - t], -1) * (de[..., None] > 0.05)
    return (rgb * 255).astype(np.uint8)


def selftest():
    fails = []
    q = float(corner(np.array(1e9), np.array(1e9)))
    if abs(q - 0.25) > 1e-6:
        fails.append("a quadrant's form factor is %r, not 1/4" % q)
    whole = float(rect_factor(-1e9, 1e9, -1e9, 1e9))
    if abs(whole - 1.0) > 1e-6:
        fails.append("the whole plane's form factor is %r, not 1" % whole)
    rng = np.random.default_rng(3)
    u = KEY_U0 + rng.random(2_000_000) * (KEY_U1 - KEY_U0)
    v = KEY_V0 + rng.random(2_000_000) * (KEY_V1 - KEY_V0)
    mc = float(np.mean(key_weight(u, v)) * (KEY_U1 - KEY_U0) * (KEY_V1 - KEY_V0))
    if abs(mc - KEY_FACTOR) > 1e-3 * KEY_FACTOR:
        fails.append("the key square's form factor %r disagrees with Monte Carlo %r" % (KEY_FACTOR, mc))
    sharma = float(ciede2000(np.array([50.0, 2.6772, -79.7751]), np.array([50.0, 0.0, -82.7485])))
    if abs(sharma - 2.0425) > 1e-4:
        fails.append("CIEDE2000 on Sharma pair 1 gives %r, not 2.0425" % sharma)
    sharma17 = float(ciede2000(np.array([50.0, 2.5, 0.0]), np.array([73.0, 25.0, -18.0])))
    if abs(sharma17 - 27.1492) > 1e-4:
        fails.append("CIEDE2000 on Sharma pair 17 gives %r, not 27.1492" % sharma17)
    empty = {"cards": [], "ground": (0.5, 0.5, 0.5), "width": 4, "height": 4}
    global scene
    real = scene
    scene = lambda theme, scale: empty
    try:
        _, d, i = trace_rows(("x", 1, 0, 4, 256, 256, 4, 0))
        s = screen_space(empty)[1]
    finally:
        scene = real
    if np.abs(d - 0.5).max() > 2e-3 or np.abs(i).max() != 0:
        fails.append("an empty canvas does not trace to its albedo: direct %r indirect %r" % (d.max(), i.max()))
    if np.abs(s - 0.5).max() > 1e-12:
        fails.append("an empty canvas does not screen-space shade to its albedo: %r" % s.max())
    flat = np.full((8, 8, 3), 0.6)
    if delta_e(flat, flat).max() != 0:
        fails.append("the instrument reports a difference between identical images")
    tinted = flat.copy()
    tinted[..., 2] += 0.02
    if delta_e(flat, tinted).max() <= JND / 2:
        fails.append("the instrument cannot see a 0.02 linear blue tint on a 0.6 grey")
    sub = flat.copy()
    sub[..., 2] += 0.0005
    if delta_e(flat, sub).max() > 0.5:
        fails.append("the instrument reports a 0.0005 tint as visible")
    furnace = {"cards": [(4.0, 12.0, 4.0, 12.0, 3.0, (1.0, 1.0, 1.0))], "ground": (1.0, 1.0, 1.0),
               "width": 16, "height": 16}
    g = globals()
    saved = {k: g[k] for k in ("L_SKY", "L_KEY", "KEY_SHARE", "scene")}
    g.update(L_SKY=1.0, L_KEY=0.0, KEY_SHARE=0.0, scene=lambda theme, scale: furnace)
    try:
        _, d, i = trace_rows(("x", 1, 0, 16, 256, 16, 60, 1))
    finally:
        g.update(saved)
    total = (d + i)[..., 0]
    rim = total[8, 13]
    if abs(total.mean() - 1.0) > 0.02 or abs(rim - 1.0) > 0.03:
        fails.append("white furnace: mean %.4f, rim %.4f, both must be 1" % (total.mean(), rim))
    if d[8, 13, 0] > 0.9:
        fails.append("white furnace: the rim's direct-only term %.4f shows no occlusion to fill" % d[8, 13, 0])
    return fails


def multibounce(ao, albedo):
    a = 2.0404 * albedo - 0.3324
    b = -4.7951 * albedo + 0.6417
    c = 2.7552 * albedo + 0.6903
    x = ao[..., None]
    return np.maximum(x, ((x * a + b) * x + c) * x)


def run(out, spp, kspp, depth, seed, workers):
    os.makedirs(out, exist_ok=True)
    report = {"invocation": " ".join(sys.argv), "spp": spp, "key_spp": kspp, "depth": depth,
              "key_centre": KEY_CENTRE, "key_half": KEY_HALF, "key_share": KEY_SHARE, "jnd": JND, "cases": []}
    for theme in ("dark", "light"):
        for scale in (1, 2):
            sc, direct, indirect = render_reference(theme, scale, spp, kspp, depth, seed, workers)
            full = direct + indirect
            ss, mb, lifted = screen_space(sc)
            ground = ~lifted
            tag = "%s_%d00" % (theme, scale)
            for name, img in (("reference", full), ("direct_only", direct), ("screen_space", ss),
                              ("screen_space_multibounce", mb)):
                save(os.path.join(out, "%s_%s.png" % (tag, name)), img)
            de_t = delta_e(full, direct)
            de_s = delta_e(full, ss)
            de_v = delta_e(direct, ss)
            de_m = delta_e(full, mb)
            Image.fromarray(heat(de_m)).save(os.path.join(out, "%s_heat_multibounce.png" % tag))
            Image.fromarray(heat(de_t)).save(os.path.join(out, "%s_heat_transport.png" % tag))
            Image.fromarray(heat(de_s)).save(os.path.join(out, "%s_heat_screen_space.png" % tag))
            row = np.concatenate([quantise(full), quantise(ss), quantise(mb)], 1)
            Image.fromarray((row * 255).astype(np.uint8)).save(os.path.join(out, "%s_pair.png" % tag))
            case = {"theme": theme, "scale": scale, "width": sc["width"], "height": sc["height"],
                    "indirect_max_linear": [round(float(x), 5) for x in indirect.reshape(-1, 3).max(0)],
                    "indirect_ground_max_linear": [round(float(x), 5) for x in indirect[ground].max(0)],
                    "reference_vs_direct_only": stats(de_t),
                    "reference_vs_direct_only_ground": stats(de_t, ground),
                    "reference_vs_screen_space": stats(de_s),
                    "direct_only_vs_screen_space": stats(de_v),
                    "reference_vs_multibounce": stats(de_m),
                    "reference_vs_multibounce_ground": stats(de_m, ground)}
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
    report["bleed_profiles"] = profiles(spp, kspp, depth, seed)
    for row in report["bleed_profiles"]:
        print(json.dumps(row), flush=True)
    with open(os.path.join(out, "gate.json"), "w") as fh:
        json.dump(report, fh, indent=1)


OUTER_SIDES = ((0, "left", -6), (1, "right", -6), (2, "left", 6), (3, "right", 6))


def profiles(spp, kspp, depth, seed, reach=34):
    rows = []
    for theme in ("light", "dark"):
        sc = scene(theme, 1)
        _, mb, _ = screen_space(sc)
        for k, side, dy in OUTER_SIDES:
            x0, x1, y0, y1, dz, _ = sc["cards"][k]
            row = int((y0 + y1) / 2) + dy
            _, d, i = trace_rows((theme, 1, row, row + 1, spp, kspp, depth, seed))
            full = d[0] + i[0]
            xs = [int(x0) - 1 - j for j in range(reach)] if side == "left" else [int(x1) + j for j in range(reach)]
            de = delta_e(full[xs][None], mb[row, xs][None])[0]
            bounce = i[0][xs].max(1)
            over = [j for j, v in enumerate(de) if v > JND]
            half = next((j for j, v in enumerate(bounce) if v < bounce[0] / 2), None)
            rows.append({"theme": theme, "card": k, "dz": dz, "side": side, "row": row,
                         "rim_bounce_linear": round(float(bounce[0]), 4), "half_decay_px": half,
                         "band_over_jnd_px": (max(over) + 1) if over else 0,
                         "max_delta_e_vs_multibounce": round(float(de.max()), 2)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="/tmp/transport_gate")
    ap.add_argument("--spp", type=int, default=1024)
    ap.add_argument("--key-spp", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    fails = selftest()
    for f in fails:
        print("SELFTEST FAIL:", f)
    if fails:
        sys.exit(1)
    print("selftest: clean")
    if not a.selftest:
        run(a.out, a.spp, a.key_spp, a.depth, a.seed, a.workers)


if __name__ == "__main__":
    main()
