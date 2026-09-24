#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import json
import os
import shutil
import hashlib

ROOT = Path(
    "/home/exx/Documents/Cyril_Kana_Python_Project/Projjet_cours_profond"
)

SRC = ROOT / "iNature1000_P2OT"

TRAIN_TXT = SRC / "iNaturalist18_train.txt"
VAL_TXT = SRC / "iNaturalist18_val.txt"

OUT500 = ROOT / "iNature500_P2OT"
OUT100 = ROOT / "iNature100_P2OT"


# ============================================================
# IO
# ============================================================

def read_txt(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rel, label = line.rsplit(maxsplit=1)
                label = int(label)
            except Exception as e:
                raise RuntimeError(
                    f"Ligne invalide {path}:{lineno}: {line!r}"
                ) from e

            rows.append((rel, label))

    return rows


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


# ============================================================
# ORIGINAL iNaturalist CATEGORY ID FROM PATH
#
# train_val2018/Plantae/7390/image.jpg
#                          ^^^^
# ============================================================

def original_category_id(rel):
    parts = Path(rel).parts

    if len(parts) < 4:
        raise RuntimeError(f"Chemin inattendu : {rel}")

    if parts[0] != "train_val2018":
        raise RuntimeError(
            f"Chemin ne commence pas par train_val2018/: {rel}"
        )

    return int(parts[-2])


# ============================================================
# FULL SOURCE VALIDATION
# ============================================================

def validate_source(train, val):

    print("=" * 72)
    print("VALIDATION iNature1000_P2OT")
    print("=" * 72)

    train_labels = {y for _, y in train}
    val_labels = {y for _, y in val}

    expected = set(range(1000))

    assert train_labels == expected, (
        f"Train labels != 0..999 "
        f"(n={len(train_labels)}, "
        f"min={min(train_labels)}, max={max(train_labels)})"
    )

    assert val_labels == expected, (
        f"Val labels != 0..999 "
        f"(n={len(val_labels)})"
    )

    # --------------------------------------------------------
    # label P2OT -> original iNaturalist category ID
    # --------------------------------------------------------

    train_mapping_sets = defaultdict(set)
    val_mapping_sets = defaultdict(set)

    for rel, label in train:
        train_mapping_sets[label].add(
            original_category_id(rel)
        )

    for rel, label in val:
        val_mapping_sets[label].add(
            original_category_id(rel)
        )

    for label in range(1000):

        if len(train_mapping_sets[label]) != 1:
            raise RuntimeError(
                f"Train label {label} correspond à plusieurs "
                f"category_id: {train_mapping_sets[label]}"
            )

        if len(val_mapping_sets[label]) != 1:
            raise RuntimeError(
                f"Val label {label} correspond à plusieurs "
                f"category_id: {val_mapping_sets[label]}"
            )

    train_map = {
        label: next(iter(train_mapping_sets[label]))
        for label in range(1000)
    }

    val_map = {
        label: next(iter(val_mapping_sets[label]))
        for label in range(1000)
    }

    assert train_map == val_map, (
        "Le mapping label P2OT -> category_id diffère "
        "entre train et val."
    )

    train_counts = Counter(y for _, y in train)
    val_counts = Counter(y for _, y in val)

    assert len(train) == 54974, len(train)
    assert len(val) == 3000, len(val)

    assert min(train_counts.values()) == 9
    assert max(train_counts.values()) == 1000

    assert set(val_counts.values()) == {3}, (
        "Validation n'a pas exactement 3 images/classe"
    )

    print(f"[OK] train images : {len(train)}")
    print(f"[OK] val images   : {len(val)}")
    print("[OK] labels       : exactement 0..999")
    print("[OK] mapping train == val")
    print("[OK] 1 label P2OT == 1 category_id original")
    print("[OK] val           : 3 images / classe")
    print(
        f"[OK] train counts : "
        f"{min(train_counts.values())} .. "
        f"{max(train_counts.values())}"
    )

    return train_map


# ============================================================
# LINK IMAGE
# ============================================================

def materialize_image(rel, out_root):

    src = SRC / rel
    dst = out_root / rel

    if not src.is_file():
        raise FileNotFoundError(src)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return "existing"

    try:
        os.link(src, dst)
        return "hardlink"

    except OSError:
        os.symlink(src.resolve(), dst)
        return "symlink"


# ============================================================
# BUILD ONE SUBSET
# ============================================================

def build_subset(
    *,
    name,
    K,
    selected_source_labels,
    train,
    val,
    source_label_to_original,
    out_root,
    expected_train,
    expected_val,
):

    print()
    print("=" * 72)
    print(f"BUILD {name}")
    print("=" * 72)

    assert len(selected_source_labels) == K
    assert len(set(selected_source_labels)) == K

    selected_set = set(selected_source_labels)

    # P2OT target_transform:
    #
    # for i, k in enumerate(include_classes):
    #     target_xform_dict[k] = i
    #
    old_to_new = {
        old: new
        for new, old in enumerate(selected_source_labels)
    }

    # Preserve original sample ordering
    train_sub = [
        (rel, old_to_new[label], label)
        for rel, label in train
        if label in selected_set
    ]

    val_sub = [
        (rel, old_to_new[label], label)
        for rel, label in val
        if label in selected_set
    ]

    assert len(train_sub) == expected_train, (
        len(train_sub), expected_train
    )

    assert len(val_sub) == expected_val, (
        len(val_sub), expected_val
    )

    assert {
        new for _, new, _ in train_sub
    } == set(range(K))

    assert {
        new for _, new, _ in val_sub
    } == set(range(K))

    val_counts = Counter(
        new for _, new, _ in val_sub
    )

    assert set(val_counts.values()) == {3}

    # --------------------------------------------------------
    # Clean output
    # --------------------------------------------------------

    if out_root.exists():
        print(f"[REMOVE] {out_root}")
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True)

    # --------------------------------------------------------
    # Link only selected images
    # --------------------------------------------------------

    unique_images = sorted(
        {rel for rel, _, _ in train_sub}
        |
        {rel for rel, _, _ in val_sub}
    )

    hardlinks = 0
    symlinks = 0

    print(
        f"[IMAGES] {len(unique_images)} images à matérialiser"
    )

    for idx, rel in enumerate(unique_images, 1):

        mode = materialize_image(
            rel,
            out_root,
        )

        if mode == "hardlink":
            hardlinks += 1
        elif mode == "symlink":
            symlinks += 1

        if idx % 10000 == 0:
            print(
                f"  {idx}/{len(unique_images)} "
                f"hardlinks={hardlinks} "
                f"symlinks={symlinks}"
            )

    # --------------------------------------------------------
    # Write P2OT-style TXT
    # --------------------------------------------------------

    train_txt = out_root / "iNaturalist18_train.txt"
    val_txt = out_root / "iNaturalist18_val.txt"

    with train_txt.open("w", encoding="utf-8") as f:
        for rel, new_label, _ in train_sub:
            f.write(
                f"{rel} {new_label}\n"
            )

    with val_txt.open("w", encoding="utf-8") as f:
        for rel, new_label, _ in val_sub:
            f.write(
                f"{rel} {new_label}\n"
            )

    # --------------------------------------------------------
    # Preserve mapping to original iNaturalist categories
    # --------------------------------------------------------

    new_to_original = {
        new: source_label_to_original[old]
        for new, old in enumerate(
            selected_source_labels
        )
    }

    mapping_file = (
        out_root /
        "P2OT_LABEL_TO_ORIGINAL_CATEGORY_ID.json"
    )

    mapping_file.write_text(
        json.dumps(
            {
                str(k): int(v)
                for k, v in new_to_original.items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    train_counts = Counter(
        new for _, new, _ in train_sub
    )

    min_count = min(train_counts.values())
    max_count = max(train_counts.values())
    natural_ir = max_count / min_count

    manifest = {
        "dataset": name,
        "construction": "exact_P2OT_recursive_subset",
        "source": str(SRC),

        "p2ot_all_class_num": [
            8142,
            2000,
            1000,
            500,
            200,
            100,
        ],

        "selection_formula":
            "upper_select_class[::interval][:num]",

        "num_classes": K,
        "num_train": len(train_sub),
        "num_val": len(val_sub),

        "train_min_class_count": min_count,
        "train_max_class_count": max_count,
        "natural_imbalance_ratio": natural_ir,

        "selected_labels_in_iNature1000_space":
            selected_source_labels,

        "label_mapping_iNature1000_to_subset": {
            str(old): int(new)
            for old, new in old_to_new.items()
        },

        "subset_label_to_original_iNaturalist_category": {
            str(new): int(original)
            for new, original in new_to_original.items()
        },

        "image_storage": {
            "hardlinks": hardlinks,
            "symlinks": symlinks,
        },

        "sha256": {
            "iNaturalist18_train.txt":
                sha256(train_txt),

            "iNaturalist18_val.txt":
                sha256(val_txt),

            "P2OT_LABEL_TO_ORIGINAL_CATEGORY_ID.json":
                sha256(mapping_file),
        },
    }

    manifest_file = (
        out_root /
        f"{name.upper()}_MANIFEST.json"
    )

    manifest_file.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[OK] {name}")
    print(f"     train      : {len(train_sub)}")
    print(f"     val        : {len(val_sub)}")
    print(f"     classes    : {K}")
    print(f"     min class  : {min_count}")
    print(f"     max class  : {max_count}")
    print(f"     IR naturel : {natural_ir:.6f}")
    print(f"     hardlinks  : {hardlinks}")
    print(f"     symlinks   : {symlinks}")
    print(f"     output     : {out_root}")


# ============================================================
# MAIN
# ============================================================

def main():

    assert SRC.is_dir(), SRC
    assert TRAIN_TXT.is_file(), TRAIN_TXT
    assert VAL_TXT.is_file(), VAL_TXT
    assert (SRC / "train_val2018").is_dir()

    train = read_txt(TRAIN_TXT)
    val = read_txt(VAL_TXT)

    source_label_to_original = validate_source(
        train,
        val,
    )

    # ========================================================
    # EXACT P2OT RECURSION
    # ========================================================

    classes1000 = list(range(1000))

    # 1000 -> 500
    interval_500 = 1000 // 500
    classes500 = (
        classes1000[::interval_500][:500]
    )

    # 500 -> 200
    interval_200 = 500 // 200
    classes200 = (
        classes500[::interval_200][:200]
    )

    # 200 -> 100
    interval_100 = 200 // 100
    classes100 = (
        classes200[::interval_100][:100]
    )

    assert classes500 == list(range(0, 1000, 2))
    assert classes200 == list(range(0, 800, 4))
    assert classes100 == list(range(0, 800, 8))

    print()
    print("[P2OT RECURSION]")
    print(
        f"1000 -> 500 : interval={interval_500}"
    )
    print(
        f"500  -> 200 : interval={interval_200}"
    )
    print(
        f"200  -> 100 : interval={interval_100}"
    )

    # ========================================================
    # BUILD 500 FIRST
    # ========================================================

    build_subset(
        name="iNature500_P2OT",
        K=500,
        selected_source_labels=classes500,
        train=train,
        val=val,
        source_label_to_original=
            source_label_to_original,
        out_root=OUT500,
        expected_train=27711,
        expected_val=1500,
    )

    # ========================================================
    # THEN BUILD 100
    # ========================================================

    build_subset(
        name="iNature100_P2OT",
        K=100,
        selected_source_labels=classes100,
        train=train,
        val=val,
        source_label_to_original=
            source_label_to_original,
        out_root=OUT100,
        expected_train=6964,
        expected_val=300,
    )

    print()
    print("=" * 72)
    print("SUCCESS — EXACT P2OT SUBSETS")
    print("=" * 72)

    print(OUT500)
    print(OUT100)


if __name__ == "__main__":
    main()
