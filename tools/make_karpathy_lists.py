#!/usr/bin/env python3
"""
make_karpathy_lists.py
----------------------

Generate file lists for Karpathy-style splits from `dataset_coco.json`.

- Works even if you only have train2014 locally.
- Creates:
    * train_only.list                   (always, if train2014 exists)
    * val5k.list / test5k.list          (only if val2014 exists)
    * restval.list                      (only if val2014 exists)
    * train_plus_restval.list           (only if both train2014 and restval exist)
- Optional: create a temporary validation split by holding out N images from train
  (useful when you don’t have val2014 yet): --holdout-n 5000

Paths inside lists are absolute, so you can use `rsync --files-from=...` directly.

Usage:
  python tools/make_karpathy_lists.py \
      --json /path/to/dataset_coco.json \
      --root /path/to/coco_root \
      --holdout-n 0

Where:
  coco_root contains train2014/ and/or val2014/ folders.
"""

import argparse
import json
import os
from collections import Counter

def build_path(cocoid: int, split: str, root: str) -> str:
    """Map cocoid + split to absolute JPEG path under coco root."""
    if split == "train":
        return os.path.join(root, "train2014", f"COCO_train2014_{int(cocoid):012d}.jpg")
    # Karpathy val/test/restval all come from val2014
    return os.path.join(root, "val2014", f"COCO_val2014_{int(cocoid):012d}.jpg")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to dataset_coco.json")
    ap.add_argument("--root", required=True, help="COCO root that contains train2014/ and/or val2014/")
    ap.add_argument("--outdir", default=None, help="Where to write the .list files (default: <root>)")
    ap.add_argument("--holdout-n", type=int, default=0,
                    help="If >0, create train_holdout.list from the last N train images (temporary val).")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = os.path.abspath(args.outdir or root)
    os.makedirs(outdir, exist_ok=True)

    with open(args.json, "r") as f:
        data = json.load(f)
    images = data["images"]

    # Inspect available folders
    has_train = os.path.isdir(os.path.join(root, "train2014"))
    has_val   = os.path.isdir(os.path.join(root, "val2014"))

    print(f"[Info] COCO root         : {root}")
    print(f"[Info] train2014 present : {has_train}")
    print(f"[Info] val2014 present   : {has_val}")

    # Split counts in JSON
    counts = Counter(im.get("split") for im in images)
    print(f"[Info] JSON split counts : {dict(counts)}")

    # Helper to dump list files safely
    def dump(split_names, filename, require_folder=True):
        """Write absolute paths for given splits to filename. Skip if folder missing."""
        if require_folder and not ((set(split_names) == {"train"} and has_train) or
                                   (set(split_names) <= {"val","test","restval"} and has_val)):
            print(f"[Skip] {filename}: required image folder not present.")
            return 0
        n = 0
        out_path = os.path.join(outdir, filename)
        with open(out_path, "w") as out:
            for im in images:
                sp = im.get("split")
                if sp in split_names:
                    cid = int(im["cocoid"])
                    out.write(build_path(cid, sp, root) + "\n")
                    n += 1
        print(f"[Write] {filename}: {n} lines")
        return n

    # Always try to write train list if train2014 exists
    n_train = dump({"train"}, "train_only.list", require_folder=True)

    # val/test/restval only if val2014 exists
    n_val   = dump({"val"},     "val5k.list", require_folder=True)
    n_test  = dump({"test"},    "test5k.list", require_folder=True)
    n_rv    = dump({"restval"}, "restval.list", require_folder=True)

    # train+restval (requires both train2014 and val2014 + restval in JSON)
    if has_train and has_val and n_rv > 0:
        out_path = os.path.join(outdir, "train_plus_restval.list")
        n = 0
        with open(out_path, "w") as out:
            for im in images:
                sp = im.get("split")
                if sp in {"train", "restval"}:
                    cid = int(im["cocoid"])
                    # restval images physically live in val2014
                    sp_for_path = "train" if sp == "train" else "val"
                    out.write(build_path(cid, sp_for_path, root) + "\n")
                    n += 1
        print(f"[Write] train_plus_restval.list: {n} lines")
    else:
        print("[Info] Skipping train_plus_restval.list (missing restval or folders).")

    # Optional: hold out N train images as a temporary validation list
    if args.holdout_n > 0 and has_train and n_train > args.holdout_n:
        # Deterministic holdout: take the last N by cocoid order
        train_ids = sorted([int(im["cocoid"]) for im in images if im.get("split") == "train"])
        holdout = train_ids[-args.holdout_n:]
        out_path = os.path.join(outdir, "train_holdout.list")
        with open(out_path, "w") as out:
            for cid in holdout:
                out.write(build_path(cid, "train", root) + "\n")
        print(f"[Write] train_holdout.list: {len(holdout)} lines (temporary val)")
        # You can also write train_minus_holdout.list if desired
        out_path2 = os.path.join(outdir, "train_minus_holdout.list")
        with open(out_path2, "w") as out:
            for cid in train_ids[:-args.holdout_n]:
                out.write(build_path(cid, "train", root) + "\n")
        print(f"[Write] train_minus_holdout.list: {len(train_ids)-args.holdout_n} lines")

    print("[Done] List generation complete.")

if __name__ == "__main__":
    main()
