# CleanCrop — find and fix bad labels in a YOLO dataset

If your detector "learns the wrong things", the cause is often in the labels:
boxes with the wrong class, boxes around background or other objects, the same
object boxed twice, or the same frames in both train and val.
CleanCrop finds these automatically and lets you fix them in a browser.

---

## Quick start (Windows)

1. **Double-click `CleanCrop.bat`** (or drag your dataset folder onto it).
   The first run installs everything it needs (a few minutes).
2. Paste the path of your dataset folder — the one containing `data.yaml`.
3. Wait for the analysis (first run on CPU: roughly 2 boxes/second; later runs reuse the cache).
4. The dashboard opens in your browser. Review the flagged boxes.
5. Press **Save Changes to Dataset**. The label `.txt` files are updated and a
   backup of every changed file is kept in `curation_output/<dataset>/label_backups/`.
6. Retrain.

Keep the black CleanCrop window open while you review; closing it stops the dashboard.
Your review decisions are auto-saved, so you can close the browser and continue later
(run `CleanCrop.bat` again and choose **[1] Continue reviewing**).

> Tip: work on a copy of your dataset, or commit it to git, so you can always compare.

---

## What it checks

| Tab | What it finds | How |
|---|---|---|
| **1. Suspected Mislabels** | Boxes whose class looks wrong (e.g. a *dog* labelled *pigeon*) | A classifier trained on the *other* folds of the data predicts every box; confident disagreement with the label is flagged. Similar boxes are shown as evidence. |
| **2. Unusual Boxes** | Boxes that don't look like the rest of their class — other species, empty background, bad crops | Local Outlier Factor inside each class |
| **3. Broken / Double Boxes** | Same object boxed twice, two classes on one object, boxes outside the image, class ids not in `data.yaml` | Read directly from the label files |
| **4. Near-Duplicates** | Almost identical boxes in different images — train↔val leaks listed first | Cosine similarity of embeddings |
| **All Boxes** | Browse/search/filter everything | — |
| **Backgrounds** | Images with no boxes (empty/missing label file), with count and % of images | Read from the label files |
| **Distribution** | Boxes/images per class, per camera (`nvr-1_NN`) and source device (part before `_nvr`), day vs night, box sizes, per-class diversity, and a golden-test readiness checklist (pass/warn/fail) | File names `<device>_nvr-1_NN_YYYYMMDD_HHMMSS_*.jpg` (day = 06-18 h; no timestamp: brightness < 70 = night); small < 32 px, large > 96 px; diversity = mean cosine distance of embeddings |

Every crop is shown with surrounding context and the box outline. **Click any crop to see the
full image** with all of its boxes drawn — essential for small objects.

### Similarity map (Overview)
- **2D** (t-SNE) and **3D** (PCA, labelled X/Y/Z axes, floor grid, depth-sized points). In 3D: drag = rotate,
  right-drag or Ctrl-drag = pan, wheel = zoom toward the cursor, double-click = centre/orbit on a point,
  orientation gizmo (click an axis or **Front / Top / Side**), **Auto-rotate**. Picking is depth-aware (the point
  nearest the camera wins) and the hovered point is highlighted.
- **Hover** shows the crop in a card by the cursor. **Click** pins a draggable window: crop + full frame (with its box),
  label, model guess, cluster, a *What to do* line, **Approve / Remove / Relabel / Undo**, and prev/next through the
  same cluster or the flagged-unreviewed list (arrow keys too). Esc closes it.
- **Shift+drag** = lasso (2D and 3D), then approve / remove / relabel all selected boxes.
- **Color: cluster** names each cluster (e.g. *C3: mostly pigeon, 12% dog – check*) and draws its outline on the
  2D map; click a cluster in the legend to show only it plus a grid of its images.
- Top bar: dataset name, **reviewed X/Y**, left to review, filters (class, camera, day/night, flagged only,
  unreviewed only; they apply to the map and the grids), **Next problem** (`N`), dark/light theme, `?` shortcut help.
  Every decision shows a toast with **Undo**. **Full screen** (`F`) keeps a side panel with status and the current box.
- **Training ready?** tab: background %, unreviewed flags, golden test size, a checklist and **Export clean dataset**
  (writes `curation_output/<name>/clean_export_<time>/` via `export_clean`; the source dataset is not touched).

### Keyboard review (Suspected Mislabels and Unusual Boxes tabs)
`A` (or `K`) approve · `X`/`Del` remove box · `1`-`9` relabel to class N · `R` accept suggested class (mislabels) ·
arrows (or `D`) next / previous · `U` undo the last keyboard or map-selection decision · `F` full image.
A counter shows **reviewed N / M** for the current thresholds.

### Sensitivity sliders
Lower values flag more boxes. Start with the defaults; lower them once the obvious cases are fixed.

---

## Supported dataset layouts

```
dataset/                         dataset/
├── data.yaml (or classes.txt)   ├── data.yaml
├── images/                      ├── train/
│   ├── train/  *.jpg            │   ├── images/
│   └── val/                     │   └── labels/
└── labels/                      └── valid/ ...
    ├── train/  *.txt
    └── val/
```

Each `.txt` line: `class_id x_center y_center width height` (normalized 0–1).

---

## Command line

```bash
python dataset_curator_yolo.py C:\path\to\dataset            # analyse + open dashboard
python dataset_curator_yolo.py C:\path\to\dataset --serve-only   # reopen dashboard, keep review progress
```

| Option | Default | Description |
|---|---|---|
| `--output-dir` | `curation_output/<dataset name>` | Where crops, cache, reports and backups go |
| `--model` | `dinov2` | Embedding model: `dinov2` or `clip` |
| `--mislabel-conf` | `0.7` | Initial mislabel confidence slider |
| `--lof-threshold` | `1.5` | Initial unusual-box slider |
| `--dup-threshold` | `0.97` | Initial duplicate slider |
| `--golden-split FRAC` | off | Also write `golden_test.txt` (image paths): ~FRAC of the images, stratified by camera × class × day/night, preferring images far apart in embedding space; near-duplicates of a chosen image go with it |
| `--no-serve` / `--no-browser` | | Only analyse / don't open the browser |
| `--no-cache` | | Recompute all embeddings |
| `--export-labeling` | | Copy flagged images + labels (YOLO `.txt` and X-AnyLabeling `.json`) into `for_labeling/` (menu option 4) |
| `--sync-labels` | | Copy labels fixed in `for_labeling/` back into the dataset, with backups (menu option 5) |
| `--export-folder` | | Save every flagged box as an annotated full image in `to_verify/` (also menu option 3 in `CleanCrop.bat`) |
| `--export-clean OUT_DIR` | | Write a **ready-to-train** YOLO dataset to a new folder: review decisions applied (deleted boxes dropped, relabels applied, images in `review_state.json` `excluded_images` skipped), background images kept, and images in `golden_test.txt` (if present) put in a separate `test` split; writes `data.yaml`, `classes.txt`, `export_summary.json`. The source dataset is not changed |
| `--port` | `8000` | Next free port is used if busy |

Outputs in the output folder:
- `curation_report.csv` — every box with its flags, most suspicious first (open in Excel)
- `curation_report.json` — full analysis used by the dashboard
- `label_backups/<timestamp>/` — original versions of label files you changed
- `cache/` — embedding cache (delete to force a full recompute)

The dashboard server only listens on `localhost`.

### Tests

```bash
pip install pytest
pytest tests -q      # tiny synthetic dataset, no embedding model needed
```

---

## License

MIT — see [LICENSE](LICENSE).

## Use from Claude (MCP)

`mcp_server.py` is a stdio MCP server that lets Claude run the analysis and review results.
Install the SDK: `pip install "mcp>=1.2"` (also in requirements.txt).

Tools: `run_analysis` (background job) / `job_status`, `get_summary`, `list_flagged` (mislabel | outlier | duplicate | issue),
`get_item`, `decide` (approve / exclude / relabel into `review_state.json`, shown by the dashboard),
`relabel_box` / `remove_box` (edit the YOLO label file, backup in `curation_output/<name>/label_backups`),
`export_for_labeling`, `sync_labels`, `export_clean(dataset_dir, out_dir)` (ready-to-train copy, see `--export-clean`),
`get_clusters`, `get_distribution`, `open_dashboard`. All paths are absolute.

**Claude Code** (add `--scope user` to make it available in every project):

```
claude mcp add --scope user curator -- python "C:\path\to\yolo-dataset-curator--main\mcp_server.py"
```

**Claude Desktop** (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "curator": {
      "command": "python",
      "args": ["C:\path\to\yolo-dataset-curator--main\mcp_server.py"]
    }
  }
}
```

Tip: reload the dashboard after `decide`, otherwise its next auto-save overwrites the decisions.
