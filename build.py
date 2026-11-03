#!/usr/bin/env python3
"""Build the cache for the BRIGHT viewer (incremental: re-run any time).

Creates under --cache (default: ./cache next to this file):
  meta.json                 tiles, events, scenes (tiles re-stitched by GeoTIFF position), split schemes
  tile_stats.json           per-tile stats, so re-runs only process new tiles
  p64/ p256/{opt,sar,gt}/   low-res proxies of every tile (optical pre-event, SAR post-event, ground truth)
  atlas/{opt,sar,gt}/       one small image per scene (32 px per tile) for the zoomed-out mosaic
  runs/<name>/              per-tile confusion matrices + prediction / error proxies of an evaluation run

Evaluation runs are found automatically: every folder under --results that has an `original/` folder of
`<tile>_building_damage.png` class maps (what bda_benchmark/script/standard_ML/infer_UNet.py writes).
Register other folders with --run NAME=DIR.

    python build.py              # everything (first run ~2 min)
    python build.py --runs-only  # just pick up new evaluation folders
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import tifffile
from PIL import Image

cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.environ.get('BRIGHT_DATASET', '/media/gigi/STOCARE/BRIGHT_DATASET')
DEFAULT_REPO = os.path.normpath(os.path.join(HERE, '..', 'BRIGHT'))
DEFAULT_RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))
DEFAULT_CACHE = os.path.join(HERE, 'cache')

TILE = 1024
LEVELS = (64, 256)
ATLAS = 32          # px per tile in the per-scene overview images
LINK = 3.0          # tiles within 3 tile-widths of each other (same event, CRS, pixel size) form one scene
SAR_GAIN = 1.5      # display stretch only: raw SAR median is ~52/255, 99th percentile ~167

CLASS_NAMES = ['background', 'intact', 'damaged', 'destroyed']
CLASS_RGB = [(0, 0, 0), (70, 181, 121), (228, 189, 139), (182, 70, 69)]   # same colours as infer_UNet.py
# error categories (index 1..5 in the error map): key, legend text, colour, opacity weight
ERR_CATS = [
    ('correct', 'building, correct class', (235, 235, 235), 0.12),
    ('missed', 'missed building (predicted background)', (239, 68, 68), 1.0),
    ('false', 'false building (background predicted as building)', (56, 189, 248), 1.0),
    ('under', 'damage under-estimated', (168, 85, 247), 1.0),
    ('over', 'damage over-estimated', (251, 146, 60), 1.0),
]

# Chen et al. 2025 (ESSD), Table 2: place, disaster type, date, GSD (m/px), optical & SAR data providers
EVENT_INFO = {
    'beirut-explosion': ('Beirut, Lebanon', 'Explosion', '04 Aug 2020', '1', 'Maxar & Capella'),
    'bata-explosion': ('Bata, Equatorial Guinea', 'Explosion', '07 Mar 2021', '0.5', 'Maxar & Capella'),
    'congo-volcano': ('Goma, DR Congo', 'Volcano eruption', '22 May 2021', '0.33', 'Maxar & Capella'),
    'haiti-earthquake': ('Les Cayes, Haiti', 'Earthquake', '14 Aug 2021', '0.48', 'Maxar & Capella'),
    'la_palma-volcano': ('La Palma, Spain', 'Volcano eruption', '19 Sep - 13 Dec 2021', '0.3-0.35', 'IGN (Spain) & Capella'),
    'marshall-wildfire': ('Boulder, USA (Marshall fire)', 'Wildfire', '30 Dec 2021 - 01 Jan 2022', '0.6', 'NAIP & Capella'),
    'ukraine-conflict': ('Ukraine', 'Armed conflict', '22 Mar - 21 Sep 2022', '0.6', 'Google Earth & Capella'),
    'turkey-earthquake': ('Turkey', 'Earthquake', '06 Feb 2023', '0.30-0.35', 'Maxar & Capella & Umbra'),
    'myanmar-hurricane': ('Kyaukpyu, Myanmar', 'Cyclone', '14 May 2023', '0.6', 'Google Earth & Capella'),
    'hawaii-wildfire': ('Maui, Hawaii, USA', 'Wildfire', '08-09 Aug 2023', '0.6', 'NOAA & Capella'),
    'morocco-earthquake': ('Morocco', 'Earthquake', '08 Sep 2023', '0.35-0.4', 'Maxar & Capella'),
    'libya-flood': ('Derna, Libya', 'Flood', '10 Sep 2023', '0.35', 'Maxar & Capella'),
    'mexico-hurricane': ('Acapulco, Mexico', 'Hurricane', '25 Oct 2023', '0.35-0.8', 'Google Earth & Capella'),
    'noto-earthquake': ('Noto, Japan', 'Earthquake', '01 Jan 2024', '0.5', 'GSI & Umbra'),
}
TYPE_ORDER = ['earthquake', 'wildfire', 'volcano', 'explosion', 'flood', 'conflict', 'hurricane']  # infer_UNet.py groups
SCHEMES = {   # name -> (folder inside the repo, {label: list file})
    'standard_ML': ('bda_benchmark/dataset/splitname/standard_ML',
                    {'train': 'train_set.txt', 'val': 'val_set.txt', 'test': 'test_set.txt'}),
    'dfc25': ('bda_benchmark/dataset/splitname/dfc25',
              {'train': 'train_set.txt', 'val': 'val_set.txt', 'test': 'test_set.txt', 'holdout': 'holdout_set.txt'}),
    'cvprw26': ('cvprw26/data/splits',
                {'train': 'train_set.txt', 'val': 'val_set.txt', 'holdout': 'holdout_set.txt',
                 'train_extra': 'train_set_extra.txt'}),
    'few_shot': ('bda_benchmark/dataset/splitname/few_shot', {'train': 'train_set.txt', 'val': 'val_set.txt'}),
}
LABEL_PRIORITY = ['test', 'holdout', 'val', 'train', 'train_extra']


# ----------------------------------------------------------------------------- small helpers
def event_of(name):
    return name.rsplit('_', 1)[0]


def type_of(event):
    return event.rsplit('-', 1)[-1]


def tile_paths(dataset, name):
    return (os.path.join(dataset, 'pre-event', f'{name}_pre_disaster.tif'),
            os.path.join(dataset, 'post-event', f'{name}_post_disaster.tif'),
            os.path.join(dataset, 'target', f'{name}_building_damage.tif'))


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, separators=(',', ':'))
    os.replace(tmp, path)


def save_image(arr, path, **kw):
    tmp = path + '.tmp'
    fmt = 'JPEG' if path.endswith('.jpg') else 'PNG'
    Image.fromarray(arr).save(tmp, format=fmt, **kw)
    os.replace(tmp, path)


def encode(arr, fmt):
    import io
    buf = io.BytesIO()
    if fmt == 'jpg':
        Image.fromarray(arr).save(buf, format='JPEG', quality=90)
    else:
        Image.fromarray(arr).save(buf, format='PNG', compress_level=1)
    return buf.getvalue()


def safe_run_name(name):
    return re.sub(r'[^\w.\-]', '_', name)


# ----------------------------------------------------------------------------- image readers / renderers
def read_pre(path):
    a = tifffile.imread(path)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    return np.ascontiguousarray(a[..., :3]).astype(np.uint8)


def read_sar(path):
    a = tifffile.imread(path)
    if a.ndim == 3:
        a = a[..., 0]
    return np.clip(a.astype(np.float32) * SAR_GAIN, 0, 255).astype(np.uint8)


def read_label(path):
    a = tifffile.imread(path)
    if a.ndim == 3:
        a = a[..., 0]
    return np.asarray(a).astype(np.uint8)


def read_pred(path):
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.uint8)


def coverage(cat_map, n_cats, size):
    """Per-category coverage fraction (0..1) of a class map, area-averaged down to size x size."""
    out = []
    for k in range(1, n_cats + 1):
        m = (cat_map == k).astype(np.float32)
        out.append(m if size == cat_map.shape[0] else cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA))
    return out


def soft_rgba(fracs, rgbs, weights):
    """Blend category colours by coverage; opacity grows with coverage (sqrt keeps small buildings visible)."""
    num = np.zeros(fracs[0].shape + (3,), np.float32)
    den = np.zeros(fracs[0].shape, np.float32)
    for f, rgb, w in zip(fracs, rgbs, weights):
        fw = f * w
        num += fw[..., None] * np.asarray(rgb, np.float32)
        den += fw
    rgb = num / np.maximum(den, 1e-6)[..., None]
    alpha = np.sqrt(np.clip(den, 0, 1)) * 255
    return np.dstack([rgb, alpha]).round().astype(np.uint8)


def class_fracs_rgba(fracs):
    return soft_rgba(fracs, CLASS_RGB[1:], [1, 1, 1])


def err_fracs_rgba(fracs):
    return soft_rgba(fracs, [c[2] for c in ERR_CATS], [c[3] for c in ERR_CATS])


def error_map(gt, pr):
    e = np.zeros(gt.shape, np.uint8)
    gb, pb = gt > 0, pr > 0
    both = gb & pb
    e[both & (gt == pr)] = 1
    e[gb & ~pb] = 2
    e[~gb & pb] = 3
    e[both & (pr < gt)] = 4
    e[both & (pr > gt)] = 5
    return e


def confusion(gt, pr):
    ok = (gt < 4) & (pr < 4)
    return np.bincount(4 * gt[ok].astype(np.int64) + pr[ok], minlength=16)[:16]


def metrics(cm16):
    """Benchmark metrics from a 4x4 confusion matrix (rows = truth), as computed in the repo's train/infer scripts."""
    m = np.asarray(cm16, np.float64).reshape(4, 4)
    tp = np.diag(m)
    iou = tp / (m.sum(0) + m.sum(1) - tp + 1e-7)
    tp_loc, fp_loc, fn_loc = m[1:, 1:].sum(), m[0, 1:].sum(), m[1:, 0].sum()
    f1_loc = 2 * tp_loc / (2 * tp_loc + fp_loc + fn_loc + 1e-7)
    f1s = []
    for k in (1, 2, 3):   # damage F1 on ground-truth building pixels, harmonic mean over the 3 classes
        tk = m[k, k]
        fn = m[k].sum() - tk
        fp = m[1:, k].sum() - tk
        f1s.append(2 * tk / (2 * tk + fp + fn + 1e-7))
    f1_clf = 0.0 if min(f1s) <= 0 else 3 / sum(1 / f for f in f1s)
    return {'oa': float(tp.sum() / max(m.sum(), 1)), 'iou': [float(x) for x in iou], 'miou': float(iou.mean()),
            'f1_loc': float(f1_loc), 'f1_clf': float(f1_clf)}


def render_full(dataset, layer, name, pred_dir=None):
    """Full-resolution layer for the detail view -> (content type, bytes)."""
    pre_p, post_p, lab_p = tile_paths(dataset, name)
    if layer == 'opt':
        return 'image/jpeg', encode(read_pre(pre_p), 'jpg')
    if layer == 'sar':
        return 'image/jpeg', encode(read_sar(post_p), 'jpg')
    if layer == 'gt':
        return 'image/png', encode(class_fracs_rgba(coverage(read_label(lab_p), 3, TILE)), 'png')
    pr = read_pred(os.path.join(pred_dir, f'{name}_building_damage.png'))
    if layer == 'pred':
        return 'image/png', encode(class_fracs_rgba(coverage(pr, 3, TILE)), 'png')
    if layer == 'err':
        return 'image/png', encode(err_fracs_rgba(coverage(error_map(read_label(lab_p), pr), 5, TILE)), 'png')
    raise ValueError(layer)


# ----------------------------------------------------------------------------- geo
def geo_info(path):
    """(EPSG code, x of top-left corner, y of top-left corner, pixel width, pixel height) from GeoTIFF tags."""
    with tifffile.TiffFile(path) as t:
        g = t.geotiff_metadata or {}
    code = g.get('ProjectedCSTypeGeoKey', g.get('GeographicTypeGeoKey'))
    code = int(getattr(code, 'value', code)) if code is not None else None
    tp, sc = g.get('ModelTiepoint'), g.get('ModelPixelScale')
    if tp is None or sc is None:
        return None
    return [code, float(tp[3] - tp[0] * sc[0]), float(tp[4] + tp[1] * sc[1]), float(sc[0]), float(sc[1])]


def utm_inverse(E, N, zone, north=True):
    """UTM easting/northing (WGS84) -> (lat, lon) degrees, Snyder's series (sub-metre accuracy)."""
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    x, y = E - 500000.0, N if north else N - 10000000.0
    mu = y / k0 / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    p1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
          + (151 * e1 ** 3 / 96) * math.sin(6 * mu) + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    n1 = a / math.sqrt(1 - e2 * math.sin(p1) ** 2)
    t1, c1 = math.tan(p1) ** 2, ep2 * math.cos(p1) ** 2
    r1 = a * (1 - e2) / (1 - e2 * math.sin(p1) ** 2) ** 1.5
    d = x / (n1 * k0)
    lat = p1 - (n1 * math.tan(p1) / r1) * (d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
                                          + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = math.radians((zone - 1) * 6 - 180 + 3) + (d - (1 + 2 * t1 + c1) * d ** 3 / 6 + (
        5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120) / math.cos(p1)
    return math.degrees(lat), math.degrees(lon)


def to_latlon(code, x, y):
    if code == 4326:
        return y, x
    if code == 3857:
        r = 6378137.0
        return math.degrees(2 * math.atan(math.exp(y / r)) - math.pi / 2), math.degrees(x / r)
    if code and 32601 <= code <= 32660:
        return utm_inverse(x, y, code - 32600, True)
    if code and 32701 <= code <= 32760:
        return utm_inverse(x, y, code - 32700, False)
    if code and 26901 <= code <= 26923:          # NAD83 / UTM north (Marshall fire)
        return utm_inverse(x, y, code - 26900, True)
    return None, None


def pixel_metres(code, sx, lat):
    if code == 4326:
        return sx * 111320.0 * math.cos(math.radians(lat))
    if code == 3857:
        return sx * math.cos(math.radians(lat))
    return sx


# ----------------------------------------------------------------------------- per-tile workers
def process_tile(job):
    dataset, cache, name = job
    pre_p, post_p, lab_p = tile_paths(dataset, name)
    pre, sar, lab = read_pre(pre_p), read_sar(post_p), read_label(lab_p)
    for layer, img in (('opt', pre), ('sar', sar)):
        small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        save_image(small, os.path.join(cache, 'p256', layer, f'{name}.jpg'), quality=85)
        save_image(cv2.resize(small, (64, 64), interpolation=cv2.INTER_AREA),
                   os.path.join(cache, 'p64', layer, f'{name}.jpg'), quality=85)
    fr = coverage(lab, 3, 256)
    save_image(class_fracs_rgba(fr), os.path.join(cache, 'p256', 'gt', f'{name}.png'))
    save_image(class_fracs_rgba([cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in fr]),
               os.path.join(cache, 'p64', 'gt', f'{name}.png'))
    return name, {'counts': np.bincount(lab.ravel(), minlength=4)[:4].tolist(),
                  'nodata': round(float(np.all(pre == 0, axis=-1).mean()), 4),
                  'geo': geo_info(pre_p)}


def process_pred(job):
    dataset, run_dir, pred_dir, name = job
    gt = read_label(tile_paths(dataset, name)[2])
    pr = read_pred(os.path.join(pred_dir, f'{name}_building_damage.png'))
    fr_p, fr_e = coverage(pr, 3, 256), coverage(error_map(gt, pr), 5, 256)
    for layer, fr, fn in (('pred', fr_p, class_fracs_rgba), ('err', fr_e, err_fracs_rgba)):
        save_image(fn(fr), os.path.join(run_dir, 'p256', layer, f'{name}.png'))
        save_image(fn([cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in fr]),
                   os.path.join(run_dir, 'p64', layer, f'{name}.png'))
    return name, confusion(gt, pr).tolist()


def executor(workers):
    return ProcessPoolExecutor(max(1, workers), mp_context=mp.get_context('spawn'))


# ----------------------------------------------------------------------------- scenes + metadata
def cluster_scenes(names, stats, events):
    """Group tiles into scenes (same event, CRS and pixel size, spatially connected) and give each tile a
    (col, row) position inside its scene, in tile units."""
    groups = {}
    for k, n in enumerate(names):
        g = stats[n]['geo']
        key = (event_of(n), g[0], f'{g[3]:.6g}', f'{g[4]:.6g}') if g else (event_of(n), 'nogeo')
        groups.setdefault(key, []).append(k)
    found = []   # (event, [tile indices], cols list, rows list)
    for key, idx in groups.items():
        if key[1] == 'nogeo':   # no georeference: lay out as a square block
            side = math.ceil(math.sqrt(len(idx)))
            found.append((key[0], idx, [j % side for j in range(len(idx))], [j // side for j in range(len(idx))], None))
            continue
        g = np.array([stats[names[k]]['geo'][1:] for k in idx])
        gx = (g[:, 0] - g[:, 0].min()) / (TILE * g[0, 2])
        gy = (g[:, 1].max() - g[:, 1]) / (TILE * g[0, 3])
        parent = list(range(len(idx)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a in range(len(idx)):
            for b in np.nonzero((np.abs(gx - gx[a]) <= LINK) & (np.abs(gy - gy[a]) <= LINK))[0]:
                ra, rb = find(a), find(int(b))
                if ra != rb:
                    parent[ra] = rb
        comps = {}
        for a in range(len(idx)):
            comps.setdefault(find(a), []).append(a)
        for comp in comps.values():
            cx, cy = gx[comp], gy[comp]
            found.append((key[0], [idx[a] for a in comp], list(cx - cx.min()), list(cy - cy.min()), key[1]))

    ev_rank = {e: r for r, e in enumerate(events)}
    found.sort(key=lambda s: (ev_rank[s[0]], -len(s[1])))
    per_event = {}
    for s in found:
        per_event[s[0]] = per_event.get(s[0], 0) + 1
    scenes, tile_scene, tile_col, tile_row = [], [0] * len(names), [0.0] * len(names), [0.0] * len(names)
    area_no = {}
    for sid, (ev, idx, cols, rows, code) in enumerate(found):
        area_no[ev] = area_no.get(ev, 0) + 1
        lat = [stats[names[k]]['lat'] for k in idx if stats[names[k]].get('lat') is not None]
        lon = [stats[names[k]]['lon'] for k in idx if stats[names[k]].get('lon') is not None]
        res = [stats[names[k]]['res'] for k in idx if stats[names[k]].get('res')]
        for k, c, r in zip(idx, cols, rows):
            tile_scene[k], tile_col[k], tile_row[k] = sid, round(float(c), 3), round(float(r), 3)
        scenes.append({
            'id': sid, 'event': ev, 'area': area_no[ev], 'n': len(idx), 'crs': code,
            'label': ev if per_event[ev] == 1 else f'{ev} · area {area_no[ev]}',
            'cols': int(math.ceil(max(cols) + 1)), 'rows': int(math.ceil(max(rows) + 1)),
            'lat': round(float(np.mean(lat)), 5) if lat else None, 'lon': round(float(np.mean(lon)), 5) if lon else None,
            'res': round(float(np.median(res)), 3) if res else None,
        })
    return scenes, tile_scene, tile_col, tile_row


def assemble_meta(names, stats, repo, dataset):
    for n in names:   # tile-centre lat/lon and pixel size in metres
        st = stats[n]
        if st.get('geo') and 'lat' not in st:
            code, x0, y0, sx, sy = st['geo']
            lat, lon = to_latlon(code, x0 + TILE / 2 * sx, y0 - TILE / 2 * sy)
            st['lat'] = round(lat, 6) if lat is not None else None
            st['lon'] = round(lon, 6) if lon is not None else None
            st['res'] = round(pixel_metres(code, sx, lat), 4) if lat is not None else None
    events = sorted({event_of(n) for n in names},
                    key=lambda e: (TYPE_ORDER.index(type_of(e)) if type_of(e) in TYPE_ORDER else 99, e))
    ev_idx = {e: k for k, e in enumerate(events)}
    names = sorted(names, key=lambda n: (ev_idx[event_of(n)], n))
    scenes, t_scene, t_col, t_row = cluster_scenes(names, stats, events)

    schemes, name_set = {}, set(names)
    for sname, (rel, files) in SCHEMES.items():
        member = {}
        for label, fn in files.items():
            p = os.path.join(repo, rel, fn)
            if os.path.isfile(p):
                with open(p) as f:
                    for line in f:
                        if line.strip():
                            member.setdefault(line.strip(), set()).add(label)
        if not member:
            continue
        labels = [l for l in LABEL_PRIORITY if l in files]
        schemes[sname] = {
            'source': rel, 'labels': labels,
            'tile': ['+'.join(l for l in LABEL_PRIORITY if l in member.get(n, ())) for n in names],
            'missing': sorted(n for n in member if n not in name_set)[:20],
        }
    ev_info = {}
    for e in events:
        place, ptype, date, gsd, src = EVENT_INFO.get(e, ('?', '?', '?', '?', '?'))
        ev_info[e] = {'type': type_of(e), 'place': place, 'paper_type': ptype, 'date': date, 'gsd': gsd,
                      'sources': src, 'n': sum(1 for n in names if event_of(n) == e)}
    return {
        'built': time.strftime('%Y-%m-%d %H:%M:%S'), 'dataset': dataset, 'tile_px': TILE, 'atlas_px': ATLAS,
        'classes': CLASS_NAMES, 'class_rgb': CLASS_RGB,
        'err_cats': [{'key': k, 'label': l, 'rgb': rgb} for k, l, rgb, _ in ERR_CATS],
        'types': [t for t in TYPE_ORDER if any(type_of(e) == t for e in events)],
        'event_list': events, 'events': ev_info, 'schemes': schemes, 'scenes': scenes,
        'tiles': {
            'name': names, 'event': [ev_idx[event_of(n)] for n in names], 'scene': t_scene, 'col': t_col, 'row': t_row,
            'counts': [stats[n]['counts'] for n in names], 'nodata': [stats[n]['nodata'] for n in names],
            'lat': [stats[n].get('lat') for n in names], 'lon': [stats[n].get('lon') for n in names],
            'res': [stats[n].get('res') for n in names],
            'crs': [stats[n]['geo'][0] if stats[n].get('geo') else None for n in names],
        },
    }


def build_atlases(meta, src_root, dst_root, layers, only=None):
    """Paste 64 px proxies (shrunk to ATLAS px) into one RGBA PNG per scene; cells without a tile stay transparent."""
    T = meta['tiles']
    by_scene = {}
    for k, s in enumerate(T['scene']):
        by_scene.setdefault(s, []).append(k)
    for layer in layers:
        ext = 'jpg' if layer in ('opt', 'sar') else 'png'
        os.makedirs(os.path.join(dst_root, 'atlas', layer), exist_ok=True)
        for sc in meta['scenes']:
            w, h = sc['cols'] * ATLAS, sc['rows'] * ATLAS
            canvas = np.zeros((h, w, 4), np.uint8)
            for k in by_scene.get(sc['id'], []):
                n = T['name'][k]
                if only is not None and n not in only:
                    continue
                im = np.array(Image.open(os.path.join(src_root, 'p64', layer, f'{n}.{ext}')).convert('RGBA'))
                im = cv2.resize(im, (ATLAS, ATLAS), interpolation=cv2.INTER_AREA)
                x, y = int(round(T['col'][k] * ATLAS)), int(round(T['row'][k] * ATLAS))
                canvas[y:y + ATLAS, x:x + ATLAS] = im[:max(0, min(ATLAS, h - y)), :max(0, min(ATLAS, w - x))]
            save_image(canvas, os.path.join(dst_root, 'atlas', layer, f'scene_{sc["id"]}.png'))


def build_base(dataset, repo, cache, workers, log=print):
    pre_dir = os.path.join(dataset, 'pre-event')
    names = sorted(f[:-len('_pre_disaster.tif')] for f in os.listdir(pre_dir) if f.endswith('_pre_disaster.tif'))
    names = [n for n in names if all(os.path.exists(p) for p in tile_paths(dataset, n))]
    for L in LEVELS:
        for layer in ('opt', 'sar', 'gt'):
            os.makedirs(os.path.join(cache, f'p{L}', layer), exist_ok=True)
    stats_path = os.path.join(cache, 'tile_stats.json')
    stats = load_json(stats_path, {})

    def complete(n):
        return n in stats and all(os.path.exists(os.path.join(cache, f'p{L}', layer, f'{n}.{ext}'))
                                  for L in LEVELS for layer, ext in (('opt', 'jpg'), ('sar', 'jpg'), ('gt', 'png')))
    todo = [n for n in names if not complete(n)]
    if todo:
        log(f'building proxies for {len(todo)} tiles with {workers} workers ...')
        t0 = time.time()
        with executor(workers) as ex:
            for k, (n, st) in enumerate(ex.map(process_tile, [(dataset, cache, n) for n in todo], chunksize=8)):
                stats[n] = st
                if (k + 1) % 500 == 0 or k + 1 == len(todo):
                    log(f'  {k + 1}/{len(todo)} tiles ({time.time() - t0:.0f} s)')
        save_json(stats_path, stats)
    meta = assemble_meta(names, stats, repo, dataset)
    save_json(stats_path, stats)
    atlas_dir = os.path.join(cache, 'atlas', 'opt')
    atlases = [f for f in os.listdir(atlas_dir) if f.endswith('.png')] if os.path.isdir(atlas_dir) else []
    if todo or len(atlases) != len(meta['scenes']):
        log(f'building scene overview images for {len(meta["scenes"])} scenes ...')
        build_atlases(meta, cache, cache, ('opt', 'sar', 'gt'))
    save_json(os.path.join(cache, 'meta.json'), meta)
    log(f'{len(names)} tiles, {len(meta["events"])} events, {len(meta["scenes"])} scenes, '
        f'split schemes: {", ".join(meta["schemes"])}')
    return meta


def discover_runs(results, extra=()):
    found = {}
    if results and os.path.isdir(results):
        for d in sorted(os.listdir(results)):
            p = os.path.join(results, d, 'original')
            if os.path.isdir(p):
                found[safe_run_name(d)] = p
    for spec in extra or ():
        name, _, path = spec.partition('=')
        if not path:
            raise SystemExit(f'--run expects NAME=DIR, got {spec!r}')
        sub = os.path.join(path, 'original')
        found[safe_run_name(name)] = sub if os.path.isdir(sub) else path
    return found


def build_runs(dataset, cache, results, extra, meta, workers, log=print):
    names = set(meta['tiles']['name'])
    built = []
    for name, pred_dir in discover_runs(results, extra).items():
        files = [f for f in os.listdir(pred_dir) if f.endswith('_building_damage.png')]
        tiles = sorted(f[:-len('_building_damage.png')] for f in files)
        tiles = [n for n in tiles if n in names]
        if not tiles:
            continue
        sig = {'pred_dir': os.path.abspath(pred_dir), 'n': len(tiles),
               'mtime': max(os.path.getmtime(os.path.join(pred_dir, f'{n}_building_damage.png')) for n in tiles)}
        run_dir = os.path.join(cache, 'runs', name)
        old = load_json(os.path.join(run_dir, 'run.json'), None)
        if old and all(old.get(k) == v for k, v in sig.items()):
            continue
        log(f'run "{name}": {len(tiles)} predicted tiles in {pred_dir}')
        for L in LEVELS:
            for layer in ('pred', 'err'):
                os.makedirs(os.path.join(run_dir, f'p{L}', layer), exist_ok=True)
        cms = {}
        with executor(workers) as ex:
            for n, cm in ex.map(process_pred, [(dataset, run_dir, pred_dir, n) for n in tiles], chunksize=8):
                cms[n] = cm
        build_atlases(meta, run_dir, run_dir, ('pred', 'err'), only=set(tiles))
        summary = metrics(np.sum([cms[n] for n in tiles], axis=0))
        save_json(os.path.join(run_dir, 'run.json'),
                  {'name': name, **sig, 'built': time.strftime('%Y-%m-%d %H:%M:%S'), 'summary': summary, 'cm': cms})
        log(f'  set-level: OA {100 * summary["oa"]:.2f}  mIoU {100 * summary["miou"]:.2f}  '
            f'IoU {" / ".join(f"{100 * x:.2f}" for x in summary["iou"])}  '
            f'F1 loc {100 * summary["f1_loc"]:.2f}  F1 clf {100 * summary["f1_clf"]:.2f}')
        built.append(name)
    return built


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default=DEFAULT_DATASET, help='folder with pre-event/ post-event/ target/')
    ap.add_argument('--repo', default=DEFAULT_REPO, help='BRIGHT repo (for the split lists)')
    ap.add_argument('--results', default=DEFAULT_RESULTS, help='folder whose sub-folders are evaluation runs')
    ap.add_argument('--cache', default=DEFAULT_CACHE)
    ap.add_argument('--run', action='append', default=[], metavar='NAME=DIR', help='register another prediction folder')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--runs-only', action='store_true', help='skip the dataset pass, only (re)build runs')
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    t0 = time.time()
    meta = load_json(os.path.join(args.cache, 'meta.json'), None) if args.runs_only else None
    if meta is None:
        meta = build_base(args.dataset, args.repo, args.cache, args.workers)
    built = build_runs(args.dataset, args.cache, args.results, args.run, meta, args.workers)
    print(f'done in {time.time() - t0:.0f} s' + (f' (new/updated runs: {", ".join(built)})' if built else ''))


if __name__ == '__main__':
    main()
