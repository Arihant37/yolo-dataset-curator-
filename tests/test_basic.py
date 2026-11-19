"""Basic tests on a tiny synthetic YOLO dataset. No embedding model is loaded."""
import os
import sys
import json

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset_curator_yolo as dcy  # noqa: E402

NAMES = ["person", "forklift"]


def _img(path, color):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path)


def _lbl(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    (root).mkdir()
    (root / "data.yaml").write_text("names: [person, forklift]\n", encoding="utf-8")
    # train: a = 2 boxes, b = 1 box + 1 broken line, c = empty label file (background)
    _img(str(root / "train/images/a.jpg"), (200, 10, 10))
    _lbl(str(root / "train/labels/a.txt"), "0 0.5 0.5 0.2 0.3\n1 0.2 0.2 0.1 0.1\n")
    _img(str(root / "train/images/b.jpg"), (10, 200, 10))
    _lbl(str(root / "train/labels/b.txt"), "1 0.4 0.4 0.2 0.2\nnot a box\n")
    _img(str(root / "train/images/c.jpg"), (10, 10, 200))
    _lbl(str(root / "train/labels/c.txt"), "")
    # valid: d = no label file at all (background), e = 1 box
    _img(str(root / "valid/images/d.jpg"), (90, 90, 90))
    _img(str(root / "valid/images/e.jpg"), (150, 150, 0))
    _lbl(str(root / "valid/labels/e.txt"), "0 0.6 0.6 0.3 0.3\n")
    return root


def test_load_class_names(dataset):
    assert dcy.load_class_names(str(dataset)) == NAMES


def test_scan_dataset_counts_and_backgrounds(dataset):
    boxes, stats, problems = dcy.scan_dataset(str(dataset), NAMES)
    assert stats["images"] == 5
    assert stats["boxes"] == len(boxes) == 4
    assert stats["background_images"] == 2
    assert stats["missing_label_files"] == 1
    bgs = {b["image"]: b for b in stats["backgrounds"]}
    assert set(bgs) == {"c.jpg", "d.jpg"}
    assert bgs["d.jpg"].get("missing_label") is True
    assert len(problems) == 1 and "Unparseable" in problems[0]["problem"]
    assert stats["splits"]["train"]["boxes"] == 3 and stats["splits"]["valid"]["boxes"] == 1
    a0 = next(b for b in boxes if b["image"] == "a.jpg" and b["line"] == 0)
    assert a0["label"] == "person" and a0["bbox"] == [0.5, 0.5, 0.2, 0.3] and not a0["issues"]


def test_scan_flags_bad_boxes(tmp_path):
    root = tmp_path / "ds2"
    _img(str(root / "images/train/x.jpg"), (0, 0, 0))
    _lbl(str(root / "labels/train/x.txt"), "0 0.5 0.5 0.4 0.4\n0 0.5 0.5 0.4 0.4\n5 0.95 0.5 0.3 0.3\n")
    boxes, stats, problems = dcy.scan_dataset(str(root), NAMES)
    assert len(boxes) == 3
    assert any("Duplicate box" in i for i in boxes[1]["issues"])
    assert any("not in data.yaml" in i for i in boxes[2]["issues"])
    assert any("outside the image" in i for i in boxes[2]["issues"])


def test_label_line_parsing():
    assert dcy._parse_box_line("1 0.5 0.5 0.2 0.2") == (1, [0.5, 0.5, 0.2, 0.2])
    assert dcy._parse_box_line("1 0.5 0.5") is None
    assert dcy._parse_box_line("x 0.5 0.5 0.2 0.2") is None
    assert dcy._parse_box_line("0 0.5 0.5 0 0.2") is None  # zero size
    # order/precision-insensitive comparison used by --sync-labels
    assert dcy._parse_label_lines("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n") == \
        dcy._parse_label_lines("1 0.10001 0.1 0.1 0.1\n0 0.5 0.5 0.2 0.2")


def _fake_analysis(dataset, out):
    """A curation_report.json as analyze() would write it, without embeddings."""
    boxes, stats, problems = dcy.scan_dataset(str(dataset), NAMES)
    items = [dict(b, index=i, path=f"crops/{b['id']}.jpg", lof=1.0, pred=b["label"], pred_conf=0.9) for i, b in enumerate(boxes)]
    os.makedirs(out, exist_ok=True)
    report = {"dataset_dir": str(dataset), "class_names": NAMES, "items": items,
              "backgrounds": stats.pop("backgrounds"), "stats": stats, "problems": problems}
    with open(os.path.join(out, "curation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f)
    return {(it["image"], it["line"]): it["id"] for it in items}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [l.split() for l in f.read().splitlines() if l.strip()]


def test_export_clean_applies_decisions(dataset, tmp_path):
    out = str(tmp_path / "curation")
    ids = _fake_analysis(dataset, out)
    state = {"deleted": [ids[("a.jpg", 1)]],                 # drop the forklift box in a
             "relabeled": {ids[("b.jpg", 0)]: "person"},      # forklift -> person in b
             "kept": [ids[("a.jpg", 0)]],
             "excluded_images": ["valid/e.jpg"]}             # skip e entirely
    with open(os.path.join(out, "review_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)
    with open(os.path.join(out, "golden_test.txt"), "w", encoding="utf-8") as f:
        f.write(str(dataset / "train/images/c.jpg") + "\n")  # background goes to the test split

    before = {p: open(p, "rb").read() for p in [str(dataset / "train/labels/a.txt"), str(dataset / "train/labels/b.txt")]}
    dst = tmp_path / "clean"
    summary = dcy.export_clean(out, str(dst))

    assert _read(dst / "train/labels/a.txt") == [["0", "0.500000", "0.500000", "0.200000", "0.300000"]]
    assert _read(dst / "train/labels/b.txt") == [["0", "0.400000", "0.400000", "0.200000", "0.200000"]]  # bad line dropped
    assert (dst / "test/images/c.jpg").exists() and _read(dst / "test/labels/c.txt") == []
    assert (dst / "valid/images/d.jpg").exists() and (dst / "valid/labels/d.txt").exists()  # background kept
    assert not (dst / "valid/images/e.jpg").exists()
    assert summary["dropped_boxes"] == 1 and summary["relabeled_boxes"] == 1 and summary["skipped_images"] == 1
    assert summary["test_images"] == 1 and summary["backgrounds"] == 2 and summary["images"] == 4
    import yaml
    data = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == NAMES and data["train"] == "train/images" and data["val"] == "valid/images" and data["test"] == "test/images"
    # the source dataset is untouched
    for p, b in before.items():
        assert open(p, "rb").read() == b


def test_export_clean_refuses_bad_targets(dataset, tmp_path):
    out = str(tmp_path / "curation")
    _fake_analysis(dataset, out)
    with pytest.raises(SystemExit):
        dcy.export_clean(out, str(dataset / "clean"))      # inside the source dataset
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x.txt").write_text("x")
    with pytest.raises(SystemExit):
        dcy.export_clean(out, str(busy))                   # not empty


def test_export_clean_cli(dataset, tmp_path):
    out = str(tmp_path / "curation")
    _fake_analysis(dataset, out)
    sys_argv = sys.argv
    try:
        sys.argv = ["dcy", str(dataset), "--output-dir", out, "--export-clean", str(tmp_path / "cli_clean")]
        dcy.main()
    finally:
        sys.argv = sys_argv
    assert (tmp_path / "cli_clean/data.yaml").exists()
    assert len(os.listdir(tmp_path / "cli_clean/train/images")) == 3


def test_clustering_and_summary():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 16)) * 5
    emb = np.vstack([c + rng.normal(scale=0.1, size=(15, 16)) for c in centers]).astype(np.float32)
    labels, k, sil = dcy.cluster_embeddings(emb)
    assert k == 3 and sil > 0.5 and len(set(labels)) == 3
    items = [{"cluster": int(c), "label": "person" if n % 15 else "forklift", "lof": 1.0} for n, c in enumerate(labels)]
    summ = dcy.cluster_summary(items)
    assert sum(s["size"] for s in summ) == 45
    assert all(s["majority"] == "person" for s in summ)
    # tiny input: one cluster, no sklearn needed
    assert dcy.cluster_embeddings(emb[:4])[1] == 1
