import os
import csv
import glob
import json
import shutil
import itertools
from argparse import ArgumentParser
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from torch.utils.data import DataLoader

from dataset.dataset import (Labeler, MAX_LEN, parse, ClipDataset, train_stats, load_data_config,
                             input_dim, has_modality, check_emg_build)
from model.transformer import Transformer
from model.ca_transformer import CA_Transformer
from model.train_model import (training_loop, predict, get_device, metrics_row, per_participant,
                               save_metrics_csv)
from plots import plot_confusion, plot_roc

GRID_KEYS = ["lr", "dropout", "n_layers", "n_heads"]
D_MODEL = 64


def files_for(files, participants):
    participants = set(participants)
    return [f for f in files if parse(f)[1] in participants]


def labels_for(files, labeler):
    return np.array([labeler(f) for f in files])


def load_grid(path):
    """Read the hyperparameter config -> list of dicts, one per combination."""
    with open(path) as fh:
        cfg = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    unknown, missing = set(cfg) - set(GRID_KEYS), set(GRID_KEYS) - set(cfg)
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    cfg = {k: v if isinstance(v, list) else [v] for k, v in cfg.items()}
    bad = [h for h in cfg["n_heads"] if D_MODEL % h]
    if bad:
        raise ValueError(f"{path}: n_heads {bad} do not divide d_model={D_MODEL}")
    return [dict(zip(GRID_KEYS, combo)) for combo in itertools.product(*(cfg[k] for k in GRID_KEYS))]


def combo_name(hp):
    return f"lr{hp['lr']:g}_do{hp['dropout']:g}_L{hp['n_layers']}_H{hp['n_heads']}"


def build_model(hp, modality):
    """Transformer for one stream (motion or emg); CA_Transformer (cross-attention) for both."""
    common = dict(d_model=D_MODEL, n_heads=hp["n_heads"], n_layers=hp["n_layers"],
                  dropout=hp["dropout"], max_len=MAX_LEN)
    if modality == "both":
        return CA_Transformer(in_dim_motion=input_dim("motion"), in_dim_emg=input_dim("emg"), **common)
    return Transformer(in_dim=input_dim(modality), **common)


def write_grid_csv(results, path):
    """Grid results, best first (validation balanced accuracy, AUC breaks ties)."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        for r in sorted(results, key=lambda r: (-r["val_balanced_acc"], -r["val_auc"])):
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and k not in GRID_KEYS else v)
                        for k, v in r.items()})


def make_splits(dataset, modality, test_size, val_size, seed, labeler, cv_folds=0):
    """Participant-level splits -> list of (train_files, val_files, test_files), one per fold.
    cv_folds=0: one split (test_size of participants for test, then val_size of the rest for validation).
    cv_folds=k: k stratified outer folds; each is the test set once, and the other participants are
    split into train / validation (val_size) as above. Every participant is tested exactly once.
    Deterministic for a given dataset, modality, seed and cv_folds, so evaluate.py can rebuild it."""
    files = sorted(glob.glob(f"{dataset}/motion/new_joint_vecs/*.npy"))
    files = [f for f in files if labeler(f) is not None]           # keep clips with a class for this task
    if not files:
        raise ValueError(f"no clips in {dataset} have a '{labeler.task}' class")
    n_all = len(files)
    files = [f for f in files if has_modality(f, modality)]        # EMG/both: skip clips with no EMG
    if len(files) < n_all:
        print(f"  {n_all - len(files)} of {n_all} clips have no EMG and are skipped")

    # All splits are over unique participants (not files), so no participant is in two sets
    prt_label = {parse(f)[1]: labeler(f) for f in files}
    participants = np.array(sorted(prt_label))
    prt_labels = np.array([prt_label[p] for p in participants])
    counts = np.bincount(prt_labels, minlength=len(labeler.names))
    print(f"{len(files)} clips, {len(participants)} participants ("
          + ", ".join(f"{n} {c}" for c, n in zip(labeler.names, counts)) + ")")

    if cv_folds:
        outer = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        test_sets = [(participants[r], prt_labels[r], participants[t]) for r, t in outer.split(participants, prt_labels)]
    else:   # fixed test set, locked away until the very end
        prt_rest, prt_test, lab_rest, _ = train_test_split(participants, prt_labels, stratify=prt_labels,
                                                           test_size=test_size, random_state=seed)
        test_sets = [(prt_rest, lab_rest, prt_test)]

    splits = []
    for k, (prt_rest, lab_rest, prt_test) in enumerate(test_sets):
        # validation set for picking the epoch and the hyperparameters
        prt_train, prt_val = train_test_split(prt_rest, stratify=lab_rest,
                                              test_size=val_size, random_state=seed)
        split = tuple(files_for(files, p) for p in (prt_train, prt_val, prt_test))
        tag = f"Fold {k + 1}/{cv_folds}: " if cv_folds else ""
        print(f"{tag}participants: train {len(prt_train)}, val {len(prt_val)}, test {len(prt_test)}; "
              f"clips: train {len(split[0])}, val {len(split[1])}, test {len(split[2])}")
        splits.append(split)
    return splits


def fold_loaders(split, stats, modality, emg_norm, labeler):
    """(train_files, val_files, test_files) -> train / val / test DataLoaders."""
    train_files, val_files, test_files = split
    make = lambda fs, **kw: DataLoader(ClipDataset(fs, stats, modality, emg_norm, labeler), **kw)
    return (make(train_files, batch_size=32, shuffle=True), make(val_files, batch_size=64),
            make(test_files, batch_size=64))


def fold_dir(base, k, cv_folds):
    """Where fold k's run lives: <base> for a single split, <base>/fold<k+1> with CV."""
    return os.path.join(base, f"fold{k + 1}") if cv_folds else base


def print_test(rows):
    for r in rows:
        print(f"  {r['model']:<28} {r['level']:<12} n={r['n']:<5} balanced acc {r['balanced_acc']:.3f}  "
              f"sensitivity {r['sensitivity']:.3f}  specificity {r['specificity']:.3f}  AUC {r['auc']:.3f}")


def evaluate_and_save(models, name, loaders, test_splits, device, out_dir, labeler):
    """Score one trained model per fold on its val and test sets; write metrics.csv, confusion matrix,
    ROC and predictions. With several folds the test predictions are pooled (each participant is in
    exactly one test fold), and the confusion matrix and ROC are drawn from the pooled predictions."""
    os.makedirs(out_dir, exist_ok=True)
    cv = len(models) > 1
    rows, probs, labels, files, fold_of = [], [], [], [], []
    for k, (model, (_, val_dl, test_dl), test_files) in enumerate(zip(models, loaders, test_splits)):
        tag = f"{name}/fold{k + 1}" if cv else name
        p, y = predict(model, test_dl, device)
        ids = [parse(f)[1] for f in test_files]
        rows += [metrics_row("val", tag, "clip", *predict(model, val_dl, device)),
                 metrics_row("test", tag, "clip", p, y),
                 metrics_row("test", tag, "participant", *per_participant(p, y, ids))]
        probs.append(p); labels.append(y); files += list(test_files); fold_of += [k + 1] * len(y)
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    if cv:
        ids = [parse(f)[1] for f in files]
        rows += [metrics_row("test", f"{name}/pooled", "clip", probs, labels),
                 metrics_row("test", f"{name}/pooled", "participant", *per_participant(probs, labels, ids))]
    summary = rows[-2:]                                   # pooled (CV) or the single test set
    print("\nTEST (threshold 0.5)")
    print_test([r for r in rows if r["split"] == "test"])

    names = labeler.names
    scope = "pooled over folds" if cv else "clips"
    cm = confusion_matrix(labels, (probs >= 0.5).astype(int), labels=[0, 1])
    plot_confusion(cm, names, os.path.join(out_dir, "confusion_matrix_test.png"),
                   title=f"Test set ({scope}): sensitivity {summary[0]['sensitivity']:.3f}, "
                         f"specificity {summary[0]['specificity']:.3f}")
    fpr, tpr, _ = roc_curve(labels, probs)
    plot_roc(fpr, tpr, roc_auc_score(labels, probs), os.path.join(out_dir, "roc_test.png"),
             title=f"ROC, test set{' (pooled over folds)' if cv else ''} ({name})", positive=names[1])
    np.savez(os.path.join(out_dir, "test_predictions.npz"), probs=probs, labels=labels,
             test_files=np.array(files), fold=np.array(fold_of))
    save_metrics_csv(rows, os.path.join(out_dir, "metrics.csv"))
    return rows


def main(args):
    grid = load_grid(args.config)
    data_cfg = load_data_config(args.data_config)
    modality = data_cfg["modality"]
    dataset = args.dataset or data_cfg.get("dataset", "../../dataset_build/gait_healthy_and_stroke")
    emg_norm = data_cfg["emg_norm"] if modality in ("emg", "both") else None
    labeler = Labeler(data_cfg["label"])
    cv_folds = data_cfg["cv_folds"]
    run_name = modality + ("_per_clip" if emg_norm == "per_clip" else "")
    if labeler.task != "stroke":
        run_name = f"{labeler.task}_{run_name}"                # e.g. results/stroke_type_both
    if cv_folds:
        run_name += f"_cv{cv_folds}"                           # e.g. results/stroke_type_both_cv5
    out_dir = args.out_dir or os.path.join("results", run_name)
    model_name = "CA_Transformer" if modality == "both" else "Transformer"
    if modality in ("emg", "both"):
        check_emg_build(dataset)

    print(f"Task: {labeler.task} ({labeler.names[0]} vs {labeler.names[1]}); modality: {modality} ({model_name}"
          f"{f', EMG normalisation {emg_norm}' if emg_norm else ''}); "
          f"{f'{cv_folds}-fold CV' if cv_folds else 'single split'}; dataset {dataset}")
    splits = make_splits(dataset, modality, args.test_size, args.val_size, args.seed, labeler, cv_folds)

    # Normalisation stats from each fold's training participants only
    # (with per-clip EMG scaling the EMG stats are stored but not used)
    stats = [train_stats(s[0], modality) for s in splits]          # per fold: {stream: (mean, std)}
    loaders = [fold_loaders(s, st, modality, emg_norm, labeler) for s, st in zip(splits, stats)]

    device = get_device()
    print(f"Training on {device}; {len(grid)} hyperparameter combination(s) from {args.config}")

    # Grid search: train on train, score on validation (mean over folds with CV)
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, hp in enumerate(grid):
        name = combo_name(hp)
        print(f"\n=== Combination {i + 1}/{len(grid)}: {name}")
        fold_rows, epochs = [], []
        for k, (split, st, (train_dl, val_dl, _)) in enumerate(zip(splits, stats, loaders)):
            if cv_folds:
                print(f"--- Fold {k + 1}/{cv_folds}")
            torch.manual_seed(args.seed)                                # same init order for every combo
            path, hist = training_loop(build_model(hp, modality), train_dl, labels_for(split[0], labeler), val_dl,
                                       epochs=args.epochs, lr=hp["lr"],
                                       out_dir=fold_dir(os.path.join(out_dir, "grid", name), k, cv_folds),
                                       extra={"hparams": hp, "modality": modality, "emg_norm": emg_norm,
                                              "model": model_name, "label": labeler.task,
                                              "cv_folds": cv_folds, "fold": k, "dataset": dataset,
                                              "stats": {s: (torch.from_numpy(m), torch.from_numpy(sd))
                                                        for s, (m, sd) in st.items()}})
            model = build_model(hp, modality).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model_state"])
            fold_rows.append(metrics_row("val", name, "clip", *predict(model, val_dl, device)))
            epochs.append(hist["best_epoch"] + 1)
        res = {**hp, "best_epoch": "/".join(map(str, epochs))}
        for m in ["balanced_acc", "auc", "sensitivity", "specificity"]:
            res[f"val_{m}"] = float(np.mean([r[m] for r in fold_rows]))     # mean over folds with CV
            if cv_folds:
                res[f"val_{m}_sd"] = float(np.std([r[m] for r in fold_rows]))
        results.append(res)
        write_grid_csv(results, os.path.join(out_dir, "grid_search.csv"))  # rewritten every combo

    best = max(results, key=lambda r: (r["val_balanced_acc"], r["val_auc"]))
    best_hp = {k: best[k] for k in GRID_KEYS}
    best_name = combo_name(best_hp)
    print(f"\nBest combination: {best_name}  (val balanced acc {best['val_balanced_acc']:.3f}"
          f"{f' +/- ' + format(best['val_balanced_acc_sd'], '.3f') if cv_folds else ''}, "
          f"AUC {best['val_auc']:.3f})")
    with open(os.path.join(out_dir, "best_hparams.json"), "w") as fh:
        json.dump(best_hp, fh, indent=2)
    best_dir = os.path.join(out_dir, "grid", best_name)
    for k in range(len(splits)):                                        # best run's files at the top level
        sfx = f"_fold{k + 1}" if cv_folds else ""
        for f in ["loss.png", "accuracy.png"] + ([] if cv_folds else ["best_model.pt"]):
            stem, ext = os.path.splitext(f)
            shutil.copy(os.path.join(fold_dir(best_dir, k, cv_folds), f), os.path.join(out_dir, stem + sfx + ext))

    # The chosen combination, scored once on test (each fold's model on its own test fold)
    models = []
    for k in range(len(splits)):
        model = build_model(best_hp, modality).to(device)
        ckpt = torch.load(os.path.join(fold_dir(best_dir, k, cv_folds), "best_model.pt"),
                          map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        models.append(model)
    evaluate_and_save(models, best_name, loaders, [s[2] for s in splits], device, out_dir, labeler)
    print(f"Plots, predictions, metrics.csv and grid_search.csv saved to {out_dir}/")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_config", type=str, default="data.json",
                        help="Data options: modality (motion | emg | both), emg_norm, label, cv_folds and dataset path")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset build (overrides 'dataset' in the data config)")
    parser.add_argument("--config", type=str, default="hparams.json",
                        help="Hyperparameter grid (lr, dropout, n_layers, n_heads)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of participants in the fixed test set (ignored with cv_folds)")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Fraction of the non-test participants used for validation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: results/<modality>")
    main(parser.parse_args())
