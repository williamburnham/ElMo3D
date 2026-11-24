"""Score the N-th ranked combination of a finished grid search on the test set.

    python evaluate.py --results results/both_per_clip --rank 3

Reads <results>/grid_search.csv, loads that combination's best_model.pt, one per fold for a
cross-validated run (task, modality, EMG normalisation, hyperparameters, cv_folds and
training-set stats are stored in the checkpoint), rebuilds the same participant splits, and writes the same outputs as main.py
(metrics.csv, confusion matrix, ROC, predictions, loss/accuracy curves) to <results>/<rank>,
e.g. results/both_per_clip/3rd. --test_size, --val_size and --seed must match the training run.
"""
import os
import csv
import shutil
from argparse import ArgumentParser
import numpy as np
import torch

from dataset.dataset import train_stats, load_data_config, Labeler
from model.train_model import get_device
from main import GRID_KEYS, combo_name, build_model, make_splits, evaluate_and_save, fold_loaders, fold_dir


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def main(args):
    with open(os.path.join(args.results, "grid_search.csv"), newline="") as fh:
        ranked = list(csv.DictReader(fh))                 # already sorted best first
    if not 1 <= args.rank <= len(ranked):
        raise ValueError(f"--rank must be 1..{len(ranked)}")
    row = ranked[args.rank - 1]
    hp = {"lr": float(row["lr"]), "dropout": float(row["dropout"]),
          "n_layers": int(row["n_layers"]), "n_heads": int(row["n_heads"])}
    name = combo_name(hp)
    run_dir = os.path.join(args.results, "grid", name)
    single = os.path.exists(os.path.join(run_dir, "best_model.pt"))
    first = torch.load(os.path.join(run_dir, "best_model.pt" if single else "fold1/best_model.pt"),
                       map_location="cpu", weights_only=True)
    modality, emg_norm = first["modality"], first["emg_norm"]
    labeler = Labeler(first.get("label", "stroke"))             # older checkpoints: Healthy vs Stroke
    cv_folds = first.get("cv_folds", 0)
    assert {k: first["hparams"][k] for k in GRID_KEYS} == hp, "checkpoint does not match grid_search.csv"
    print(f"Rank {args.rank}: {name} ({labeler.task}, {modality}, EMG normalisation {emg_norm}, "
          f"{f'{cv_folds}-fold CV' if cv_folds else 'single split'}), "
          f"val balanced acc {row['val_balanced_acc']}")

    # dataset: --dataset, else the one recorded in the checkpoint, else the data config's (older checkpoints)
    dataset = args.dataset or first.get("dataset") or load_data_config(args.data_config).get("dataset")
    splits = make_splits(dataset, modality, args.test_size, args.val_size, args.seed, labeler, cv_folds)

    device = get_device()
    models, loaders = [], []
    for k, split in enumerate(splits):
        ckpt = torch.load(os.path.join(fold_dir(run_dir, k, cv_folds), "best_model.pt"),
                          map_location="cpu", weights_only=True)
        # Check the rebuilt split is the training run's: its training-set stats must match the checkpoint's
        stats = {s: (m.numpy(), sd.numpy()) for s, (m, sd) in ckpt["stats"].items()}
        for s, (m, sd) in train_stats(split[0], modality).items():
            if not (np.allclose(m, stats[s][0], atol=1e-5) and np.allclose(sd, stats[s][1], atol=1e-5)):
                raise RuntimeError(f"{s} stats differ from the checkpoint (fold {k + 1}): the split does not "
                                   "match the training run (check --dataset, --test_size, --val_size, --seed)")
        model = build_model(hp, modality).to(device)
        model.load_state_dict(ckpt["model_state"])
        models.append(model)
        loaders.append(fold_loaders(split, stats, modality, emg_norm, labeler))
    print("Split check: training-set stats match the checkpoint" + (" in every fold" if cv_folds else ""))

    out_dir = os.path.join(args.results, ordinal(args.rank))
    rows = evaluate_and_save(models, name, loaders, [s[2] for s in splits], device, out_dir, labeler)
    val_mean = np.mean([r["balanced_acc"] for r in rows if r["split"] == "val"])
    if abs(val_mean - float(row["val_balanced_acc"])) > 1e-3:
        print(f"WARNING: val balanced acc {val_mean:.4f} differs from grid_search.csv")
    for k in range(len(splits)):
        sfx = f"_fold{k + 1}" if cv_folds else ""
        for f in ["loss.png", "accuracy.png"] + ([] if cv_folds else ["best_model.pt"]):
            stem, ext = os.path.splitext(f)
            shutil.copy(os.path.join(fold_dir(run_dir, k, cv_folds), f), os.path.join(out_dir, stem + sfx + ext))
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="A main.py output folder, e.g. results/both")
    parser.add_argument("--rank", type=int, default=3, help="Rank in grid_search.csv (1 = best)")
    parser.add_argument("--data_config", type=str, default="data.json", help="Its 'dataset' path is used for checkpoints that do not record one")
    parser.add_argument("--dataset", type=str, default=None, help="Overrides the data config's dataset path")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
