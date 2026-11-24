import torch
from torch.utils.data import Dataset
import numpy as np
import os
import sys
import json
import csv

# Reuse build_dataset.py's Mean/Std logic (HumanML3D block-shared Std, unmeasured joints excluded)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from build_dataset import Moments, UNMEASURED, joint_dims, motion_stats

MAX_LEN = 196                  # tokens; one token = one 20 fps motion frame
CLASSES = {"Hlth": 0, "Strk": 1}
MODALITIES = ["motion", "emg", "both"]
METADATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/ElMo3D_data/participant_information.csv"))
EMG_NORMS = ["global", "per_clip"]   # train-set stats per channel | each channel z-scored within its own clip
EMG_CHANNELS = 20
EMG_PER_FRAME = 5              # processed EMG is 100 Hz, motion 20 fps -> 5 samples per frame


def parse(path):
    """
    Parse the label, id, action and clip from the file path.
    """
    name = os.path.basename(path)[:-4]
    # Cut first character from mirrored clip names
    if name.startswith("M"):
        name = name[1:]
    dsid, cls, action, pid, clip = name.split("-")
    return CLASSES.get(cls), f"{dsid}-{pid}", action, clip  # label, id, action, clip


class Labeler:
    """Maps a clip file to its class index, or None to leave the clip out.

    task='stroke':      Healthy (0) vs Stroke (1), from the file name (Hlth / Strk); other classes left out.
    task='stroke_type': Ischaemic (0) vs Haemorrhagic (1), from the Stroke_type column of
                        participant_information.csv; participants without a stroke type are left out.
    Class 1 is the positive class: 'sensitivity' and the ROC are reported for it."""

    TASKS = {"stroke": ["Hlth", "Strk"], "stroke_type": ["Ischaemic", "Haemorrhagic"]}

    def __init__(self, task="stroke", metadata=METADATA):
        if task not in self.TASKS:
            raise ValueError(f"label must be one of {list(self.TASKS)}, got {task!r}")
        self.task = task
        self.names = self.TASKS[task]
        if task == "stroke_type":
            with open(metadata, newline="", encoding="utf-8-sig") as fh:
                # participant id as in file names: '<sub-dataset>-<4-digit number>', e.g. '50ST-0003'
                self.meta = {f"{r['Dataset']}-{int(r['Participant_Number']):04d}": r["Stroke_type"]
                             for r in csv.DictReader(fh)}

    def __call__(self, path):
        if self.task == "stroke":
            return parse(path)[0]
        value = self.meta.get(parse(path)[1])
        return self.names.index(value) if value in self.names else None


def load_data_config(path):
    """Read the data config -> dict with 'modality', 'emg_norm', 'label', 'cv_folds' (and 'dataset' if given)."""
    with open(path) as fh:
        cfg = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    if cfg.get("modality") not in MODALITIES:
        raise ValueError(f"{path}: 'modality' must be one of {MODALITIES}, got {cfg.get('modality')!r}")
    cfg.setdefault("emg_norm", "global")
    cfg.setdefault("label", "stroke")
    cfg.setdefault("cv_folds", 0)
    if not (isinstance(cfg["cv_folds"], int) and (cfg["cv_folds"] == 0 or cfg["cv_folds"] >= 2)):
        raise ValueError(f"{path}: 'cv_folds' must be 0 (single train/val/test split) or an integer >= 2")
    if cfg["label"] not in Labeler.TASKS:
        raise ValueError(f"{path}: 'label' must be one of {list(Labeler.TASKS)}, got {cfg['label']!r}")
    if cfg["emg_norm"] not in EMG_NORMS:
        raise ValueError(f"{path}: 'emg_norm' must be one of {EMG_NORMS}, got {cfg['emg_norm']!r}")
    return cfg


def emg_path(motion_file):
    """<build>/motion/new_joint_vecs/X.npy -> <build>/emg/X.npy (same name in the build)."""
    build = os.path.dirname(os.path.dirname(os.path.dirname(motion_file)))
    return os.path.join(build, "emg", os.path.basename(motion_file))


def check_emg_build(dataset):
    """EMG / both modes need a build made with --emg processed (100 Hz envelopes, .npy)."""
    info_path = os.path.join(dataset, "build_info.json")
    emg = json.load(open(info_path)).get("emg") if os.path.exists(info_path) else None
    if emg not in (None, "processed"):
        raise ValueError(f"{dataset} was built with --emg {emg}; the classifier needs --emg processed")


STREAMS = {"motion": ["motion"], "emg": ["emg"], "both": ["motion", "emg"]}


def input_dim(stream):
    return {"motion": 263, "emg": EMG_CHANNELS * EMG_PER_FRAME}[stream]


def load_raw(motion_file, stream):
    """Un-normalised features, one row per time step: motion [T, 263] or EMG [N, 20] at 100 Hz."""
    if stream == "motion":
        return np.load(motion_file)
    return np.load(emg_path(motion_file)).T


def recorded_channels(emg):
    """A channel that is all zero in a file was not recorded (build_dataset.py convention)."""
    return np.any(emg != 0, axis=0)


def stream_stats(files, stream):
    """Mean/Std of one stream over the given (training) files, computed exactly as build_dataset.py does.
    Motion: per dimension, Std shared per HumanML3D block, unmeasured joints excluded.
    EMG: per channel, over recorded channels only; never-recorded channels left unscaled."""
    if stream == "motion":
        m = Moments(263)
        masks = {g: joint_dims(j) for g, j in UNMEASURED.items()}
        for f in files:
            x = np.load(f)
            valid = np.ones(x.shape[1], bool)
            group = parse(f)[1].split("-")[0]             # sub-dataset id
            if group in masks:
                valid[masks[group]] = False
            m.add(x, valid)
        return motion_stats(m)

    m = Moments(EMG_CHANNELS)
    for f in files:
        x = load_raw(f, "emg")
        m.add(x, recorded_channels(x))
    mean, std = m.result()
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def train_stats(files, modality):
    """{stream: (mean, std)} for each stream the modality uses, from the given (training) files."""
    return {s: stream_stats(files, s) for s in STREAMS[modality]}


def has_modality(motion_file, modality):
    return modality == "motion" or os.path.exists(emg_path(motion_file))


def motion_tokens(x, mean, std):
    return (x - mean) / std                               # [T, 263]


def emg_tokens(x, mean, std, emg_norm):
    """[N, 20] at 100 Hz -> normalised [T, 5 * 20]: the 5 samples of each 20 fps frame form one token."""
    x = x[:MAX_LEN * EMG_PER_FRAME]                       # only what the model sees
    rec = recorded_channels(x)
    if emg_norm == "per_clip":                            # each channel against itself, within the clip
        mean, std = x.mean(0), x.std(0)
        x = (x - mean) / np.where(std < 1e-8, 1.0, std)
    else:
        x = (x - mean) / std
    x[:, ~rec] = 0.0                                      # unrecorded channel stays 0 (= the mean)
    T = len(x) // EMG_PER_FRAME
    return x[:T * EMG_PER_FRAME].reshape(T, EMG_PER_FRAME * EMG_CHANNELS)


def pad(x):
    """Pad or truncate to MAX_LEN -> (features [MAX_LEN, D], padding mask [MAX_LEN], True = padding)."""
    T = min(len(x), MAX_LEN)
    out = np.zeros((MAX_LEN, x.shape[1]), np.float32)
    out[:T] = x[:T]
    pad_mask = np.ones(MAX_LEN, bool); pad_mask[:T] = False
    return torch.from_numpy(out), torch.from_numpy(pad_mask)


class ClipDataset(Dataset):
    """One clip -> (features, padding mask, label).
    motion / emg: features [MAX_LEN, input_dim], mask [MAX_LEN].
    both: features (motion, emg) and masks (motion, emg), token t = frame t in both.
    `files` are motion file paths; for EMG the matching file in <build>/emg/ is read.
    `stats` is {stream: (mean, std)} from train_stats. emg_norm='global' uses the EMG stats;
    'per_clip' z-scores each EMG channel over the part of the clip the model sees.
    `labeler` maps a file to its class (default: Healthy vs Stroke from the file name)."""

    def __init__(self, files, stats, modality, emg_norm="global", labeler=None):
        self.files = files # all file paths
        self.stats = stats
        self.streams = STREAMS[modality]
        self.emg_norm = emg_norm
        self.labeler = labeler or Labeler("stroke")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        xs, masks = [], []
        for s in self.streams:
            x = load_raw(f, s)
            x = motion_tokens(x, *self.stats[s]) if s == "motion" else emg_tokens(x, *self.stats[s], self.emg_norm)
            x, m = pad(x)
            xs.append(x); masks.append(m)
        label = self.labeler(f)
        if len(xs) == 1:
            return xs[0], masks[0], label
        return tuple(xs), tuple(masks), label
