"""
Preprocess the TUH EEG Seizure Corpus (TUSZ) for REVE fine-tuning.

Expected raw layout (official TUSZ train / dev / eval splits, patient-disjoint):
    <root>/train/<patient>/<session>/<montage>/<recording>.edf (+ .csv_bi, .csv)
    <root>/dev/...
    <root>/eval/...

Output layout (one pair of files per recording):
    <out>/train/<recording>_X.npy   float32, shape (n_windows, 16, duration*fs)   [volts]
    <out>/train/<recording>_y.npy   int64,   shape (n_windows,)
    <out>/val/...   (from TUSZ "dev")
    <out>/test/...  (from TUSZ "eval")

Example:
    python preprocess_tusz.py --root /data/TUSZ/v2.0.3/edf --task binary --workers 16
"""
import argparse
import os
import random
import re
import zlib
from multiprocessing import Pool

import mne
import numpy as np

mne.set_log_level("ERROR")

# 16-channel "double banana" TCP montage.
# Order and names are identical to the `electrodes` list in the yaml config.
BIPOLAR_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]
NEEDED_ELECTRODES = sorted({e for pair in BIPOLAR_PAIRS for e in pair})

# our split name -> TUSZ folder name
SPLIT_DIRS = {"train": "train", "val": "dev", "test": "eval"}

# binary detection uses the .csv_bi files, multiclass seizure typing uses the .csv files
ANN_EXT = {"binary": ".csv_bi", "multiclass": ".csv"}

# v2.0.5 re-annotated the corpus, so check the label set actually present in your .csv files
# (see the grep command in the notes) and override with --classes if it differs.
DEFAULT_SEIZURE_TYPES = ["fnsz", "gnsz", "cpsz", "absz", "spsz", "tcsz", "tnsz", "mysz"]


def build_label_map(task, seizure_types):
    if task == "binary":
        return {"bckg": 0, "seiz": 1}
    return {"bckg": 0, **{c: i for i, c in enumerate(seizure_types, start=1)}}

# "EEG FP1-REF" / "EEG FP1-LE" / "EEG FP1-AR"  ->  "FP1"
CH_RE = re.compile(r"^EEG\s+([A-Z0-9]+)-(REF|LE|AR)", re.IGNORECASE)


def normalize_channel(name):
    m = CH_RE.match(name.strip())
    return m.group(1).upper() if m else None


def read_annotation(path, n_samples, fs, label_map):
    """Turn a TUSZ annotation file into one label per sample (background = 0).

    Rows look like: channel,start_time,stop_time,label,confidence
    The channel column is ignored: a sample is "seizure" if any row covers it.
    """
    labels = np.zeros(n_samples, dtype=np.int64)
    unknown = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("channel"):
                continue
            parts = line.split(",")
            start, stop, lab = float(parts[1]), float(parts[2]), parts[3].strip().lower()
            if lab not in label_map:
                unknown.add(lab)
                continue
            if label_map[lab] == 0:
                continue
            s = max(0, int(round(start * fs)))
            e = min(n_samples, int(round(stop * fs)))
            labels[s:e] = label_map[lab]
    return labels, unknown


def window_labels(sample_labels, win, n_classes, thresh):
    """One label per non-overlapping window.

    A window is a seizure window if >= `thresh` of its samples are seizure;
    its class is the seizure type covering most samples. Otherwise background.
    """
    n_win = len(sample_labels) // win
    segs = sample_labels[: n_win * win].reshape(n_win, win)
    counts = np.stack([(segs == c).sum(axis=1) for c in range(n_classes)], axis=1)
    seiz_frac = counts[:, 1:].sum(axis=1) / win
    seiz_type = counts[:, 1:].argmax(axis=1) + 1
    return np.where(seiz_frac >= thresh, seiz_type, 0).astype(np.int64)


def process_recording(task):
    edf_path, ann_path, out_dir, split, cfg = task
    stem = os.path.splitext(os.path.basename(edf_path))[0]
    label_map = cfg["label_map"]
    n_classes = len(set(label_map.values()))
    counts = np.zeros(n_classes, dtype=np.int64)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")

        # map normalized names ("FP1") -> original names ("EEG FP1-REF" / "EEG FP1-LE")
        name_map = {}
        for orig in raw.ch_names:
            norm = normalize_channel(orig)
            if norm and norm not in name_map:
                name_map[norm] = orig
        missing = [e for e in NEEDED_ELECTRODES if e not in name_map]
        if missing:
            return stem, f"missing channels {missing}", counts

        raw.pick([name_map[e] for e in NEEDED_ELECTRODES])
        raw.load_data()

        sfreq = raw.info["sfreq"]
        if cfg["filter"]:
            if cfg["notch"] and sfreq / 2 > cfg["notch"]:
                raw.notch_filter(cfg["notch"])
            raw.filter(l_freq=cfg["l_freq"], h_freq=min(cfg["h_freq"], sfreq / 2 - 1))
        raw.resample(cfg["fs"])

        # bipolar derivations
        idx = {normalize_channel(n): i for i, n in enumerate(raw.ch_names)}
        data = raw.get_data()  # volts
        bipolar = np.stack([data[idx[a]] - data[idx[b]] for a, b in BIPOLAR_PAIRS])
        bipolar = bipolar.astype(np.float32)  # (16, n_samples)

        n_samples = bipolar.shape[1]
        win = int(cfg["duration"] * cfg["fs"])
        n_win = n_samples // win
        if n_win == 0:
            return stem, "recording shorter than one window", counts

        sample_labels, unknown = read_annotation(ann_path, n_samples, cfg["fs"], label_map)
        y = window_labels(sample_labels, win, n_classes, cfg["seiz_thresh"])
        X = bipolar[:, : n_win * win].reshape(16, n_win, win).transpose(1, 0, 2)

        # optional: thin out background windows (never for the test split)
        keep_frac = {"train": cfg["bckg_keep_frac"], "val": cfg["val_bckg_keep_frac"]}.get(split, 1.0)
        if keep_frac < 1.0:
            rng = np.random.default_rng(zlib.crc32(stem.encode()))
            keep = (y != 0) | (rng.random(len(y)) < keep_frac)
            X, y = X[keep], y[keep]

        if len(y) == 0:
            return stem, "no windows kept", counts

        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{stem}_X.npy"), np.ascontiguousarray(X))
        np.save(os.path.join(out_dir, f"{stem}_y.npy"), y)
        counts = np.bincount(y, minlength=n_classes)
        msg = "ok" if not unknown else f"ok (ignored unknown labels {sorted(unknown)})"
        return stem, msg, counts
    except Exception as e:  # corrupt EDF, bad header, ...
        return stem, f"error: {e}", counts


def find_recordings(split_root, task):
    items, no_ann, dups, seen = [], [], [], set()
    for dirpath, dirnames, files in os.walk(split_root):
        dirnames.sort()  # deterministic order
        for f in sorted(files):
            if not f.endswith(".edf"):
                continue
            edf = os.path.join(dirpath, f)
            ann = os.path.join(dirpath, f[:-4] + ANN_EXT[task])
            if f[:-4] in seen:  # same recording name in two montage folders would overwrite output
                dups.append(edf)
            elif os.path.exists(ann):
                seen.add(f[:-4])
                items.append((edf, ann))
            else:
                no_ann.append(edf)
    return items, no_ann, dups


def has_seizure(ann_path):
    with open(ann_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("channel"):
                continue
            parts = line.strip().split(",")
            if len(parts) > 3 and parts[3].strip().lower() != "bckg":
                return True
    return False


def subsample(recs, n, seed=0):
    """Pick n recordings for a trial run: about half with seizures, half without."""
    if n <= 0 or n >= len(recs):
        return recs
    flags = [has_seizure(ann) for _, ann in recs]
    pos = [r for r, fl in zip(recs, flags) if fl]
    neg = [r for r, fl in zip(recs, flags) if not fl]
    n_pos = min(len(pos), (n + 1) // 2)
    n_neg = min(len(neg), n - n_pos)
    n_pos = min(len(pos), n - n_neg)
    rng = random.Random(seed)
    return rng.sample(pos, n_pos) + rng.sample(neg, n_neg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="TUSZ edf root containing train/ dev/ eval/")
    p.add_argument("--out", default=None, help="output dir (default: <root>/processed_tusz)")
    p.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--fs", type=int, default=200, help="target sampling rate")
    p.add_argument("--duration", type=float, default=5, help="window length in seconds")
    p.add_argument("--seiz-thresh", type=float, default=0.8,
                   help="fraction of a window that must be seizure to label it seizure")
    p.add_argument("--bckg-keep-frac", type=float, default=1.0,
                   help="fraction of background windows kept in the TRAIN split")
    p.add_argument("--val-bckg-keep-frac", type=float, default=1.0,
                   help="same for the VAL (dev) split; changes the class balance of your val metrics")
    p.add_argument("--classes", nargs="+", default=DEFAULT_SEIZURE_TYPES,
                   help="seizure type labels for --task multiclass (class ids follow this order)")
    p.add_argument("--l-freq", type=float, default=0.5)
    p.add_argument("--h-freq", type=float, default=99.5)
    p.add_argument("--notch", type=float, default=60, help="0 disables the notch filter")
    p.add_argument("--no-filter", action="store_true", help="skip all filtering (resample only)")
    p.add_argument("--max-files", type=int, default=0,
                   help="trial run: process only this many recordings PER SPLIT "
                        "(about half with seizures); 0 = everything. Use a separate --out folder.")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    out_root = args.out or os.path.join(args.root, "processed_tusz_multiclass")
    label_map = build_label_map(args.task, args.classes)
    cfg = dict(
        task=args.task, fs=args.fs, duration=args.duration, seiz_thresh=args.seiz_thresh,
        bckg_keep_frac=args.bckg_keep_frac, val_bckg_keep_frac=args.val_bckg_keep_frac,
        l_freq=args.l_freq, h_freq=args.h_freq, notch=args.notch, filter=not args.no_filter,
        label_map=label_map,
    )

    n_classes = len(set(label_map.values()))
    print(f"label map: {label_map}")
    problems = []
    for split in args.splits:
        split_root = os.path.join(args.root, SPLIT_DIRS[split])
        if not os.path.isdir(split_root):
            raise SystemExit(f"Folder not found: {split_root}\n"
                             "Check --root (a leading '/' means an absolute path from the filesystem root).")
        recs, no_ann, dups = find_recordings(split_root, args.task)
        problems += [f"{split}\t{e}\tno annotation file" for e in no_ann]
        problems += [f"{split}\t{e}\tduplicate recording name, skipped" for e in dups]
        out_dir = os.path.join(out_root, split)
        print(f"[{split}] {len(recs)} recordings ({len(no_ann)} without annotations, {len(dups)} duplicates)")
        if args.max_files > 0:
            recs = subsample(recs, args.max_files)
            print(f"[{split}] trial run: using {len(recs)} recordings")

        tasks = [(edf, ann, out_dir, split, cfg) for edf, ann in recs]
        total = np.zeros(n_classes, dtype=np.int64)
        with Pool(processes=args.workers) as pool:
            for i, (stem, status, counts) in enumerate(pool.imap_unordered(process_recording, tasks), 1):
                total += counts
                if status != "ok":
                    problems.append(f"{split}\t{stem}\t{status}")
                if i % 50 == 0 or i == len(tasks):
                    print(f"[{split}] {i}/{len(tasks)} done")
        print(f"[{split}] windows per class: {total.tolist()}")

    os.makedirs(out_root, exist_ok=True)
    log = os.path.join(out_root, "tusz-process-error-files.txt")
    with open(log, "w") as f:
        f.write("\n".join(problems) + ("\n" if problems else ""))
    print(f"Done. {len(problems)} skipped/flagged recordings logged to {log}")


if __name__ == "__main__":
    main()
