"""
Preprocess the TUH EEG Seizure Corpus (TUSZ) for LaBraM fine-tuning.
Mirrors LaBraM's TUAB preprocessing (one pickle per window, signal in microvolts),
but reads the official TUSZ train / dev / eval folders and the seizure annotations.

Expected raw layout (patient-disjoint official splits):
    <root>/train/<patient>/<session>/<montage>/<recording>.edf (+ .csv_bi, .csv)
    <root>/dev/...      -> saved as val
    <root>/eval/...     -> saved as test

Output layout:
    <out>/train/<recording>_<window>_y<label>.pkl     {"X": float32 (n_channels, duration*fs) in uV, "y": int}
    <out>/val/...
    <out>/test/...
    <out>/meta.json                                    channels, classes, task (read by the fine-tuning code)

The label is also encoded in the file name (_y<label>) so class counts / samplers
can be built without opening any pickle.

Example:
    python preprocess_tusz_labram.py --root /data/TUSZ/v2.0.6 --task binary --workers 16
    python preprocess_tusz_labram.py --root /data/TUSZ/v2.0.6 --task multiclass --out /data/TUSZ_labram_mc
"""
import argparse
import json
import os
import pickle
import random
import re
import zlib
from multiprocessing import Pool

import mne
import numpy as np

mne.set_log_level("ERROR")

# Referential (monopolar) 10-20 channels. LaBraM learns one embedding per electrode NAME,
# so use referential channels, NOT the bipolar TCP montage used for REVE.
# These names must exist in `standard_1020` in LaBraM's utils.py.
DEFAULT_CHANNELS = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]

SPLIT_DIRS = {"train": "train", "val": "dev", "test": "eval"}
ANN_EXT = {"binary": ".csv_bi", "multiclass": ".csv"}
DEFAULT_SEIZURE_TYPES = ["fnsz", "gnsz", "cpsz", "absz", "spsz", "tcsz", "tnsz", "mysz"]

# "EEG FP1-REF" / "EEG FP1-LE" / "EEG FP1-AR" -> "FP1"
CH_RE = re.compile(r"^EEG\s+([A-Z0-9]+)-(REF|LE|AR)", re.IGNORECASE)


def build_label_map(task, seizure_types):
    if task == "binary":
        return {"bckg": 0, "seiz": 1}
    return {"bckg": 0, **{c: i for i, c in enumerate(seizure_types, start=1)}}


def normalize_channel(name):
    m = CH_RE.match(name.strip())
    return m.group(1).upper() if m else None


def read_annotation(path, n_samples, fs, label_map):
    """TUSZ annotation file -> one label per sample (background = 0).
    Rows: channel,start_time,stop_time,label,confidence. The channel column is ignored."""
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
    """One label per non-overlapping window: seizure if >= thresh of its samples are seizure,
    with the seizure type covering most samples; otherwise background (0)."""
    n_win = len(sample_labels) // win
    segs = sample_labels[: n_win * win].reshape(n_win, win)
    counts = np.stack([(segs == c).sum(axis=1) for c in range(n_classes)], axis=1)
    seiz_frac = counts[:, 1:].sum(axis=1) / win
    seiz_type = counts[:, 1:].argmax(axis=1) + 1
    return np.where(seiz_frac >= thresh, seiz_type, 0).astype(np.int64)


def process_recording(job):
    edf_path, ann_path, out_dir, split, cfg = job
    stem = os.path.splitext(os.path.basename(edf_path))[0]
    n_classes = cfg["n_classes"]
    channels = cfg["channels"]
    counts = np.zeros(n_classes, dtype=np.int64)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")

        name_map = {}
        for orig in raw.ch_names:
            norm = normalize_channel(orig)
            if norm and norm not in name_map:
                name_map[norm] = orig
        missing = [c for c in channels if c not in name_map]
        if missing:
            return stem, f"missing channels {missing}", counts

        wanted = [name_map[c] for c in channels]
        raw.pick(wanted)
        raw.reorder_channels(wanted)          # fixed channel order for every recording
        raw.load_data()

        sfreq = raw.info["sfreq"]
        raw.filter(l_freq=cfg["l_freq"], h_freq=min(cfg["h_freq"], sfreq / 2 - 1))
        if cfg["notch"] and sfreq / 2 > cfg["notch"]:
            raw.notch_filter(cfg["notch"])
        raw.resample(cfg["fs"])
        X_all = raw.get_data(units="uV").astype(np.float32)   # (C, n_samples), microvolts

        n_samples = X_all.shape[1]
        win = int(cfg["duration"] * cfg["fs"])
        n_win = n_samples // win
        if n_win == 0:
            return stem, "recording shorter than one window", counts

        sample_labels, unknown = read_annotation(ann_path, n_samples, cfg["fs"], cfg["label_map"])
        y = window_labels(sample_labels, win, n_classes, cfg["seiz_thresh"])
        X = X_all[:, : n_win * win].reshape(len(channels), n_win, win).transpose(1, 0, 2)

        keep = np.ones(n_win, dtype=bool)
        keep_frac = {"train": cfg["bckg_keep_frac"], "val": cfg["val_bckg_keep_frac"]}.get(split, 1.0)
        if keep_frac < 1.0:                    # thin background windows (never for test)
            rng = np.random.default_rng(zlib.crc32(stem.encode()))
            keep &= (y != 0) | (rng.random(n_win) < keep_frac)
        if cfg["train_max_abs_uv"] > 0 and split == "train":   # optional artifact rejection, train only
            keep &= np.abs(X).max(axis=(1, 2)) <= cfg["train_max_abs_uv"]

        idx = np.where(keep)[0]
        if len(idx) == 0:
            return stem, "no windows kept", counts

        os.makedirs(out_dir, exist_ok=True)
        for i in idx:
            lab = int(y[i])
            with open(os.path.join(out_dir, f"{stem}_{i}_y{lab}.pkl"), "wb") as f:
                pickle.dump({"X": np.ascontiguousarray(X[i]), "y": lab}, f, protocol=4)

        counts = np.bincount(y[idx], minlength=n_classes)
        msg = "ok" if not unknown else f"ok (ignored unknown labels {sorted(unknown)})"
        return stem, msg, counts
    except Exception as e:  # corrupt EDF, bad header, ...
        return stem, f"error: {e}", counts


def find_recordings(split_root, task):
    items, no_ann, dups, seen = [], [], [], set()
    for dirpath, dirnames, files in os.walk(split_root):
        dirnames.sort()
        for f in sorted(files):
            if not f.endswith(".edf"):
                continue
            edf = os.path.join(dirpath, f)
            ann = os.path.join(dirpath, f[:-4] + ANN_EXT[task])
            if f[:-4] in seen:     # same name in two montage folders would overwrite output
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
    """Trial run: n recordings, about half with seizures."""
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
    p.add_argument("--out", default=None, help="output dir (default: <root>/processed_tusz_labram)")
    p.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--fs", type=int, default=200, help="LaBraM expects 200 Hz")
    p.add_argument("--duration", type=float, default=5,
                   help="window length in whole seconds (LaBraM patch = 1 s; max 16 s)")
    p.add_argument("--seiz-thresh", type=float, default=0.5)
    p.add_argument("--bckg-keep-frac", type=float, default=1.0, help="fraction of background windows kept in TRAIN")
    p.add_argument("--val-bckg-keep-frac", type=float, default=1.0, help="same for VAL (changes val class balance)")
    p.add_argument("--classes", nargs="+", default=DEFAULT_SEIZURE_TYPES,
                   help="seizure type labels for --task multiclass (class ids follow this order)")
    p.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS,
                   help="referential channel names, must exist in LaBraM's standard_1020")
    p.add_argument("--l-freq", type=float, default=0.1)
    p.add_argument("--h-freq", type=float, default=75.0)
    p.add_argument("--notch", type=float, default=60, help="60 Hz for US-recorded TUH data; 0 disables")
    p.add_argument("--train-max-abs-uv", type=float, default=0,
                   help="optional: drop TRAIN windows whose max |amplitude| exceeds this many uV (0 = off)")
    p.add_argument("--max-files", type=int, default=0,
                   help="trial run: recordings PER SPLIT (about half with seizures); use a separate --out")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    if args.duration != int(args.duration) or not (1 <= args.duration <= 16):
        raise SystemExit("--duration must be a whole number of seconds between 1 and 16 "
                         "(LaBraM uses 1 s patches and has 16 time-position embeddings).")
    if args.fs != 200:
        print("WARNING: LaBraM's patch size is 200 samples = 1 s only at 200 Hz.")

    channels = [c.upper() for c in args.channels]
    out_root = args.out or os.path.join(args.root, "processed_tusz_labram")
    label_map = build_label_map(args.task, args.classes)
    n_classes = len(set(label_map.values()))
    class_names = ["bckg", "seiz"] if args.task == "binary" else ["bckg"] + list(args.classes)

    cfg = dict(fs=args.fs, duration=args.duration, seiz_thresh=args.seiz_thresh,
               bckg_keep_frac=args.bckg_keep_frac, val_bckg_keep_frac=args.val_bckg_keep_frac,
               l_freq=args.l_freq, h_freq=args.h_freq, notch=args.notch,
               train_max_abs_uv=args.train_max_abs_uv, label_map=label_map,
               n_classes=n_classes, channels=channels)

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(dict(task=args.task, classes=class_names, n_classes=n_classes, channels=channels,
                       fs=args.fs, duration=args.duration, unit="uV"), f, indent=2)
    print(f"label map: {label_map}\nchannels ({len(channels)}): {channels}")

    problems = []
    for split in args.splits:
        split_root = os.path.join(args.root, SPLIT_DIRS[split])
        if not os.path.isdir(split_root):
            raise SystemExit(f"Folder not found: {split_root}")
        recs, no_ann, dups = find_recordings(split_root, args.task)
        problems += [f"{split}\t{e}\tno annotation file" for e in no_ann]
        problems += [f"{split}\t{e}\tduplicate recording name, skipped" for e in dups]
        print(f"[{split}] {len(recs)} recordings ({len(no_ann)} without annotations, {len(dups)} duplicates)")
        if args.max_files > 0:
            recs = subsample(recs, args.max_files)
            print(f"[{split}] trial run: using {len(recs)} recordings")

        out_dir = os.path.join(out_root, split)
        jobs = [(edf, ann, out_dir, split, cfg) for edf, ann in recs]
        total = np.zeros(n_classes, dtype=np.int64)
        with Pool(processes=args.workers) as pool:
            for i, (stem, status, counts) in enumerate(pool.imap_unordered(process_recording, jobs), 1):
                total += counts
                if status != "ok":
                    problems.append(f"{split}\t{stem}\t{status}")
                if i % 50 == 0 or i == len(jobs):
                    print(f"[{split}] {i}/{len(jobs)} done")
        print(f"[{split}] windows per class {dict(zip(class_names, total.tolist()))}")

    log = os.path.join(out_root, "tusz-labram-process-error-files.txt")
    with open(log, "w") as f:
        f.write("\n".join(problems) + ("\n" if problems else ""))
    print(f"Done. {len(problems)} skipped/flagged recordings logged to {log}")


if __name__ == "__main__":
    main()
