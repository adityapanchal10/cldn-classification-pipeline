import itertools

from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score

from src.training import make_weighted_ce, train_classifier

def compute_full_val_metrics(
    val_targets,
    val_probs,
    class_names={0: "barrier", 1: "cation", 2: "anion"},
):
    """
    Compute full multi-class metrics for final training mode
    (all classes present in the val set).
    """
    val_targets = np.array(val_targets)
    val_probs   = np.array(val_probs)
    val_preds   = val_probs.argmax(axis=1)
    n_classes   = val_probs.shape[1]
    all_labels  = list(range(n_classes))

    acc = (val_preds == val_targets).mean() * 100

    macro_f1 = f1_score(
        val_targets, val_preds,
        average='macro', labels=all_labels, zero_division=0,
    )

    # AUC-ROC (requires >= 2 classes present)
    n_present = len(np.unique(val_targets))
    if n_present >= 2:
        try:
            import warnings
            from sklearn.exceptions import UndefinedMetricWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                auc_val = roc_auc_score(
                    val_targets, val_probs,
                    multi_class="ovr", average="macro", labels=all_labels,
                )
        except ValueError:
            auc_val = float('nan')
    else:
        auc_val = float('nan')

    # Per-class recall
    per_class_recall = {}
    for cid, cname in class_names.items():
        mask = val_targets == cid
        if mask.any():
            per_class_recall[cname] = (val_preds[mask] == cid).mean() * 100
        else:
            per_class_recall[cname] = float('nan')

    return {
        "acc":              acc,
        "macro_f1":         macro_f1,
        "auc":              auc_val,
        "per_class_recall": per_class_recall,
    }

def make_grouped_mixed_splits(file_meta):
    """
    Build grouped holdout splits:
      test = [1 barrier, 1 cation, 1 anion]
      train = all remaining files
    """
    barrier_files = [idx for idx, info in file_meta.items() if info["label"] == 0]
    cation_files  = [idx for idx, info in file_meta.items() if info["label"] == 1]
    anion_files   = [idx for idx, info in file_meta.items() if info["label"] == 2]

    splits = []
    for b, c, a in itertools.product(barrier_files, cation_files, anion_files):
        test_idx = [b, c, a]
        train_idx = [idx for idx in sorted(file_meta.keys()) if idx not in test_idx]

        splits.append({
            "test_file_idx": test_idx,
            "train_file_idx": train_idx,
            "test_file_names": [file_meta[i]["fname"] for i in test_idx],
            "train_file_names": [file_meta[i]["fname"] for i in train_idx],
        })

    return splits

def build_grouped_train_loader(
    train_file_idx,
    embeddings_by_file,
    file_meta,
    batch_size=64,
    num_classes=3,
    useWeightedSampler=False,
):
    """
    Training uses original per-file embeddings, i.e. each train MSA remains
    in its original embedding context.
    """
    train_emb_list = []
    train_lbl_list = []
    train_pid_list = []

    for file_idx in train_file_idx:
        emb   = embeddings_by_file[file_idx]
        label = file_meta[file_idx]["label"]
        nseq  = emb.shape[0]

        train_emb_list.append(emb)
        train_lbl_list.extend([label] * nseq)
        train_pid_list.extend([file_idx] * nseq)

    train_emb = torch.cat(train_emb_list, dim=0)
    train_lbl = torch.tensor(train_lbl_list, dtype=torch.long)
    train_pid = torch.tensor(train_pid_list, dtype=torch.long)

    train_ds = TensorDataset(train_emb, train_lbl, train_pid)

    fold_counts = Counter(train_lbl.numpy())
    counts_list = [fold_counts.get(i, 1) for i in range(num_classes)]

    if useWeightedSampler:
        class_sample_weights = {
            cls: 1.0 / count for cls, count in fold_counts.items() if count > 0
        }
        sample_weights = torch.tensor(
            [class_sample_weights[int(lbl)] for lbl in train_lbl.tolist()],
            dtype=torch.float,
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    return train_loader, counts_list, train_emb, train_lbl, train_pid
    
def build_grouped_val_loader(
    val_file_idx,
    embeddings_by_file,
    file_meta,
    batch_size=64,
    num_classes=3,
):
    val_emb_list = []
    val_lbl_list = []
    val_pid_list = []

    for file_idx in val_file_idx:
        emb = embeddings_by_file[file_idx]
        label = file_meta[file_idx]["label"]
        nseq  = emb.shape[0]
        seq_ids = file_meta[file_idx]["ids"]
        fnames = file_meta[file_idx]["fname"]

        val_emb_list.append(emb)
        val_lbl_list.extend([label] * nseq)
        val_pid_list.extend([file_idx] * nseq)

    val_emb = torch.cat(val_emb_list, dim=0)
    val_lbl = torch.tensor(val_lbl_list, dtype=torch.long)
    val_pid = torch.tensor(val_pid_list, dtype=torch.long)

    val_ds = TensorDataset(val_emb, val_lbl, val_pid)
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False), val_lbl, seq_ids, fnames

def compute_grouped_holdout_metrics(
    y_true,
    y_prob,
    class_names={0: "barrier", 1: "cation", 2: "anion"},
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    acc = accuracy_score(y_true, y_pred) * 100
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    ) * 100

    metrics = {
        "accuracy": acc,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "recall_barrier": per_class_recall[0],
        "recall_cation": per_class_recall[1],
        "recall_anion": per_class_recall[2],
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }
    return metrics

def compute_composition_recovery(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_counts = np.array([(y_true == c).sum() for c in [0, 1, 2]])
    pred_counts = np.array([(y_pred == c).sum() for c in [0, 1, 2]])

    true_props = true_counts / true_counts.sum()
    pred_props = pred_counts / pred_counts.sum()

    abs_prop_error = np.abs(true_props - pred_props)
    mean_abs_prop_error = abs_prop_error.mean()

    comp_df = pd.DataFrame({
        "class_idx": [0, 1, 2],
        "true_count": true_counts,
        "pred_count": pred_counts,
        "true_prop": true_props,
        "pred_prop": pred_props,
        "abs_prop_error": abs_prop_error,
    })

    return comp_df, mean_abs_prop_error

def train_grouped_holdout_cv(
    file_meta,
    embeddings_by_file,
    model_fn,
    seq_len,
    num_epochs=100,
    device='cuda',
    patience=25,
    batch_size=64,
    num_classes=3,
    class_names={0: "barrier", 1: "cation", 2: "anion"},
    useWeightedSampler=False,
    weighted_ce_power=1.0,
    warmup_epochs=5,
    optimizer_lr= 1e-3,
    optimizer_weight_decay=1e-2,
    schedular_patience=10,
    checkpoint_dir="grouped_holdout_checkpoints",
):
    """
    Grouped mixed-MSA holdout evaluation.

    Train on remaining original MSAs.
    Test on 1 held-out barrier + 1 held-out cation + 1 held-out anion,
    recomputed jointly in mixed-MSA context.
    """
    splits = make_grouped_mixed_splits(file_meta)
    grouped_results = []
    last_history = None

    useWeightedCE = False if weighted_ce_power < 0 else True

    print("\n" + "=" * 90)
    print(f"GROUPED MIXED HOLDOUT EVALUATION")
    print(f"  Folds        : {len(splits)}")
    print("=" * 90)

    for fold_idx, split in enumerate(splits, start=1):
        print("\n" + "=" * 90)
        print(f"Grouped Fold {fold_idx}/{len(splits)}")
        print(f"  Train files: {split['train_file_names']}")
        print(f"  Test files : {split['test_file_names']}")
        print("=" * 90)

        # Train loader from original train-MSA embeddings
        train_loader, counts_list, train_emb, train_lbl, train_pid = build_grouped_train_loader(
            split["train_file_idx"],
            embeddings_by_file,
            file_meta,
            batch_size=batch_size,
            num_classes=num_classes,
            useWeightedSampler=useWeightedSampler,
        )

        print(f"  Train class dist: {Counter(train_lbl.numpy())}")

        val_loader, val_lbl, test_seq_ids, test_fnames = build_grouped_val_loader(
            split["test_file_idx"],
            embeddings_by_file,
            file_meta,
            batch_size=batch_size,
        )

        print(f"  Val class dist: {Counter(val_lbl.numpy())}")

        # Fresh model and criterion 
        model = model_fn().to(device)
        if useWeightedCE:
            criterion = make_weighted_ce(counts_list, device, power=weighted_ce_power)
        else:
            criterion = None

        # Train using your existing train_classifier 
        # grouped test contains all classes, so use final-mode:
        # held_out_class=None => primary metric = macro_f1
        history = train_classifier(
            classifier=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            num_classes=num_classes,
            device=device,
            patience=patience,
            criterion=criterion,
            held_out_class=None,
            held_out_class_name=None,
            held_out_file_name=",".join(split["test_file_names"]),
            class_names=class_names,
            checkpoint_path=f"{checkpoint_dir}/grouped_holdout_fold{fold_idx}.pt" if checkpoint_dir else None,
            warmup_epochs=warmup_epochs,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
            schedular_patience=schedular_patience
        )
        last_history = history

        # Extract best-epoch validation predictions from history 
        best_epoch_idx = history["saved_epoch"] - 1
        y_true_best = np.asarray(history["val_targets"][best_epoch_idx])
        y_prob_best = np.asarray(history["val_probs"][best_epoch_idx])

        metrics = compute_grouped_holdout_metrics(
            y_true=y_true_best,
            y_prob=y_prob_best,
            class_names=class_names,
        )

        comp_df, mean_abs_prop_error = compute_composition_recovery(
            y_true_best,
            metrics["y_pred"]
        )

        pred_df = pd.DataFrame({
            "fold": fold_idx,
            "seq_id": test_seq_ids,
            "source_file": test_fnames,
            "true_label": y_true_best,
            "pred_label": metrics["y_pred"],
            "prob_barrier": y_prob_best[:, 0],
            "prob_cation": y_prob_best[:, 1],
            "prob_anion": y_prob_best[:, 2],
            "confidence": y_prob_best.max(axis=1),
            "test_files": ",".join(split["test_file_names"]),
        })

        grouped_results.append({
            "fold": fold_idx,
            "train_files": split["train_file_names"],
            "test_files": split["test_file_names"],
            "saved_epoch": history["saved_epoch"],
            "accuracy": metrics["accuracy"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "recall_barrier": metrics["recall_barrier"],
            "recall_cation": metrics["recall_cation"],
            "recall_anion": metrics["recall_anion"],
            "mean_abs_prop_error": mean_abs_prop_error,
            "confusion_matrix": metrics["confusion_matrix"],
            "composition_df": comp_df,
            "pred_df": pred_df,
            "history": history,
        })

        print("\n  Sequence-level metrics:")
        print(f"    Accuracy      : {metrics['accuracy']:.2f}%")
        print(f"    Macro Recall  : {metrics['macro_recall']:.2f}%")
        print(f"    Macro F1      : {metrics['macro_f1']:.2f}%")
        print(f"    Recall barrier: {metrics['recall_barrier']:.2f}%")
        print(f"    Recall cation : {metrics['recall_cation']:.2f}%")
        print(f"    Recall anion  : {metrics['recall_anion']:.2f}%")
        print(f"    Mean abs prop error: {mean_abs_prop_error:.4f}")
        print("\n  Confusion matrix:")
        print(metrics["confusion_matrix"])
        print("\n  Composition recovery:")
        print(comp_df)

    return grouped_results, last_history

def print_grouped_holdout_summary(grouped_results):
    print("\n" + "=" * 120)
    print("GROUPED MIXED HOLDOUT SUMMARY")
    print("=" * 120)

    if len(grouped_results) == 0:
        print("No grouped results found.")
        return

    print("-" * 120)

    print(f"{'Fold':<5} {'Test files':<45} {'Acc%':>8} {'MacroRec%':>10} {'MacroF1%':>10} "
          f"{'Barrier%':>10} {'Cation%':>10} {'Anion%':>10} {'MAPE':>10}")
    print("-" * 120)

    for r in grouped_results:
        print(f"{r['fold']:<5} "
              f"{','.join(r['test_files']):<45} "
              f"{r['accuracy']:>8.2f} "
              f"{r['macro_recall']:>10.2f} "
              f"{r['macro_f1']:>10.2f} "
              f"{r['recall_barrier']:>10.2f} "
              f"{r['recall_cation']:>10.2f} "
              f"{r['recall_anion']:>10.2f} "
              f"{r['mean_abs_prop_error']:>10.4f}")

    print("-" * 120)

    metrics_df = pd.DataFrame([{
        "accuracy": r["accuracy"],
        "macro_recall": r["macro_recall"],
        "macro_f1": r["macro_f1"],
        "recall_barrier": r["recall_barrier"],
        "recall_cation": r["recall_cation"],
        "recall_anion": r["recall_anion"],
        "mean_abs_prop_error": r["mean_abs_prop_error"],
    } for r in grouped_results])

    print("\nMean grouped-holdout metrics:")
    print(metrics_df.mean())

    print("\nStd grouped-holdout metrics:")
    print(metrics_df.std())

def save_grouped_holdout_results(grouped_results, prefix="grouped_holdout"):
    if len(grouped_results) == 0:
        print("No grouped results to save.")
        return

    summary_df = pd.DataFrame([{
        "fold": r["fold"],
        "train_files": ",".join(r["train_files"]),
        "test_files": ",".join(r["test_files"]),
        "saved_epoch": r["saved_epoch"],
        "accuracy": r["accuracy"],
        "macro_recall": r["macro_recall"],
        "macro_f1": r["macro_f1"],
        "recall_barrier": r["recall_barrier"],
        "recall_cation": r["recall_cation"],
        "recall_anion": r["recall_anion"],
        "mean_abs_prop_error": r["mean_abs_prop_error"],
    } for r in grouped_results])

    preds_df = pd.concat([r["pred_df"] for r in grouped_results], ignore_index=True)

    summary_path = f"{prefix}/summary.csv"
    preds_path   = f"{prefix}/predictions.csv"

    summary_df.to_csv(summary_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    print(f"Saved summary to: {summary_path}")
    print(f"Saved predictions to: {preds_path}")