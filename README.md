# BRIGHT viewer

Local web app for looking at the BRIGHT dataset and at evaluation results. Tiles are re-stitched into their
original scenes using the GeoTIFF coordinates, so you can see where train / val / test tiles sit, what each
event looks like, and where a model goes wrong.

The browser only loads small proxy images (64 px and 256 px per tile, plus one 32 px-per-tile overview image per
scene: ~370 MB cache instead of the 21 GB dataset). Full-resolution 1024 px tiles are read from the dataset on
demand, only for the tile you open or the few tiles on screen when zoomed all the way in.

## Run

```bash
python build.py     # once (~90 s): proxies, scenes, split lists, and any evaluation runs in ../results
python serve.py     # then open http://127.0.0.1:8765
```

Defaults match this machine (`/media/gigi/STOCARE/BRIGHT_DATASET`, `../BRIGHT`, `../results`); override with
`--dataset`, `--repo`, `--results`, `--cache` (see `--help`). Both scripts only need numpy, opencv, Pillow and
tifffile, which the base conda environment already has.

## On another machine

1. Have the dataset (e.g. from HuggingFace) in the usual `pre-event/ post-event/ target/` layout.
2. Keep this folder next to a clone of the BRIGHT repo (`../BRIGHT`, where the split lists live), or pass `--repo`.
3. `pip install numpy opencv-python pillow tifffile` if they are missing.
4. Build once with your dataset path; `serve.py` then reuses the path recorded by the build:

   ```bash
   BRIGHT_DATASET=/path/to/BRIGHT python build.py     # or: python build.py --dataset /path/to/BRIGHT
   python serve.py                                    # --port 8766 if 8765 is taken
   ```

Don't copy someone else's `cache/`: it is tied to their paths, and `build.py` makes a fresh one in a minute or two.
Evaluation runs go in `../results/<name>/original/` (or pass `--results` to both scripts).

## Adding evaluation results

Any folder under `../results/<name>/` that contains `original/<tile>_building_damage.png` class maps is a run
(this is what `bda_benchmark/script/standard_ML/infer_UNet.py --output_dir ../results/<name>` writes). After a new
evaluation, press **Rescan runs** in the page (or run `python build.py --runs-only`), then pick the run top right.
Other folders: `python build.py --run NAME=/path/to/folder`.

For each run the viewer computes per-tile confusion matrices, so the **Selection** panel shows set-level
mIoU / OA / F1 (same definitions as the repo's scripts) and the event-averaged mIoU for whatever you have
filtered, e.g. one event, one scene, or only the tiles with lots of damage. For the official U-Net checkpoint on the full test
set it reproduces the paper exactly: OA 95.47, mIoU 64.94, F1 loc 87.97, F1 clf 72.24.

## What is on screen

- **Mosaic**: every scene stitched back together; tiles that fail the filter are dimmed. **Grid**: only the
  matching tiles, ordered by *Grid order* (e.g. worst tile mIoU of the selected run).
- **Base image**: optical pre-event or SAR post-event (SAR is shown with a fixed 1.5x display stretch).
- **Overlays**: ground truth, prediction, errors (missed building, false building, damage under- or
  over-estimated), and split colour for any of the repo's split schemes (standard_ML, dfc25, cvprw26, few_shot).
- **Click a tile** for the full-resolution view: optical, SAR, truth, prediction and errors side by side with
  synced zoom, plus the tile's class counts and confusion matrix. `←` / `→` step through the current order.
- The view state lives in the URL, so a link reproduces what you are looking at.

Shortcuts: `m` mosaic/grid, `o`/`s`/`n` base image, `g`/`p`/`e`/`t` overlays, `f` fit, `?` help.

## Notes

- The server listens on localhost only. `cache/` is a derived copy of the dataset, and the imagery keeps its
  original licences (they vary per event, e.g. CC BY-NC 4.0 for the Maxar tiles), so keep it out of git and
  don't publish it.
- `cache/` can be deleted at any time; `build.py` regenerates it.
