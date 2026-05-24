from collections import Counter

import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
)

from src.training import make_weighted_ce, train_classifier, compute_full_val_metrics

def compute_honest_val_metrics(
    val_targets,
    val_probs,
    held_out_class,
    class_names={0: "barrier", 1: "cation", 2: "anion"},
):
    """
    Compute metrics valid when val set has only one class (LOFO mode).

    Returns dict with: acc, recall, correct_class_conf, confusion_str,
                       entropy, confidence_level, n_val
    """
    val_targets = np.array(val_targets)
    val_probs   = np.array(val_probs)
    val_preds   = val_probs.argmax(axis=1)
    n_classes   = val_probs.shape[1]

    # Accuracy
    acc = (val_preds == val_targets).mean() * 100

    # Recall for the held-out class
    mask_true = val_targets == held_out_class
    recall = (
        (val_preds[mask_true] == held_out_class).mean() * 100
        if mask_true.any() else 0.0
    )

    # Confusion breakdown — what did the model predict instead?
    wrong_preds   = val_preds[val_preds != held_out_class]
    confusion     = Counter(wrong_preds.tolist())
    confusion_str = ", ".join(
        f"{class_names[k]}({v})" for k, v in sorted(confusion.items())
    ) if confusion else "none"

    # Mean confidence assigned to the correct class
    correct_class_conf = val_probs[:, held_out_class].mean() * 100

    # Entropy of mean prediction distribution
    mean_probs       = val_probs.mean(axis=0)
    entropy          = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
    max_entropy      = np.log(n_classes)
    confidence_level = "decisive" if entropy < 0.5 * max_entropy else "uncertain"

    return {
        "acc":                acc,
        "recall":             recall,
        "correct_class_conf": correct_class_conf,
        "confusion_str":      confusion_str,
        "entropy":            entropy,
        "confidence_level":   confidence_level,
        "n_val":              len(val_targets),
    }


def print_epoch_summary(
    epoch,
    num_epochs,
    train_loss,
    train_acc,
    val_loss,
    val_all_targets,
    val_all_probs,
    patience_counter,
    patience,
    improved,
    held_out_class=None,
    held_out_class_name=None,
    held_out_file_name=None,
    class_names={0: "barrier", 1: "cation", 2: "anion"},
):
    """
    Print a clean, honest per-epoch summary.

    Adapts automatically based on training mode:
      - LOFO fold   (held_out_class is int)  : single-class honest metrics
      - Final train (held_out_class is None) : full multi-class metrics
    """
    is_lofo = held_out_class is not None

    val_targets = np.array(val_all_targets)
    val_probs   = np.array(val_all_probs)
    val_preds   = val_probs.argmax(axis=1)
    val_acc     = (val_preds == val_targets).mean() * 100

    mode_label = f"LOFO — held out: {held_out_file_name}" if is_lofo else "Final Training"

    print(f"\n{'-'*62}")
    print(f"  Epoch {epoch}/{num_epochs}  |  {mode_label}")
    print(f"{'-'*62}")
    print(f"  [Train]  Loss: {train_loss:.4f}  |  Acc: {train_acc:.1f}%")
    print(f"  [Val]    Loss: {val_loss:.4f}  |  Acc: {val_acc:.1f}%")
    print()

    if is_lofo:
        # LOFO mode
        m = compute_honest_val_metrics(
            val_targets, val_probs, held_out_class, class_names,
        )
        print(f"  What this fold tests: can the model identify '{held_out_class_name}'")
        print(f"  from a paralog it has never seen before?")
        print(f"  Val set: {held_out_file_name}({m['n_val']}) sequences — ALL labelled '{held_out_class_name}'")
        print()
        print(f"  Recall on '{held_out_file_name}':          {m['recall']:.1f}%")
        print(f"    → Correctly identified {m['recall']:.1f}% of {held_out_file_name} sequences")
        print()
        print(f"  Avg confidence in '{held_out_file_name}':  {m['correct_class_conf']:.1f}%")
        print(f"    → Model is {m['confidence_level']} (prediction entropy: {m['entropy']:.3f})")
        print()
        if m['confusion_str'] != "none":
            print(f"  Classes the model mistakes '{held_out_file_name}' for")
            print(f"    → Confused with: {m['confusion_str']}")
        else:
            print(f"  No misclassifications this epoch ✓")

        primary_metric = m['recall']

    else:
        # Final training mode
        m = compute_full_val_metrics(val_targets, val_probs, class_names)

        print(f"  Val set: {len(val_targets)} sequences across all classes")
        print()
        print(f"  Macro F1:  {m['macro_f1']:.4f}")
        auc_str = f"{m['auc']:.4f}" if not np.isnan(m['auc']) else "n/a"
        print(f"  AUC-ROC:   {auc_str}")
        print()
        print(f"  Per-class recall:")
        for cname, rec in m['per_class_recall'].items():
            if np.isnan(rec):
                print(f"    {cname:<10}  n/a (class not in val set)")
            else:
                filled = int(rec // 10)
                bar    = "█" * filled + "░" * (10 - filled)
                print(f"    {cname:<10} [{bar}]  {rec:.1f}%")

        primary_metric = m['macro_f1']

    print()
    metric_label = "recall" if is_lofo else "macro F1"
    metric_fmt   = f"{primary_metric:.1f}%" if is_lofo else f"{primary_metric:.4f}"
    if improved:
        print(f"  ✅ Best {metric_label} improved → {metric_fmt}  — checkpoint saved")
    else:
        print(f"  ⏳ No improvement. Patience: {patience_counter}/{patience}")

    return primary_metric

def build_loaders(
    train_idx, val_idx, all_embeddings, all_labels, all_file_idx,
    batch_size=64, num_classes=3, useWeightedSampler=False,
):
    """Build train/val DataLoaders for one LOFO fold."""
    train_emb = all_embeddings[train_idx]
    val_emb   = all_embeddings[val_idx]
    train_lbl = all_labels[train_idx]
    val_lbl   = all_labels[val_idx]
    train_pid = torch.tensor(all_file_idx[train_idx], dtype=torch.long)
    val_pid   = torch.tensor(all_file_idx[val_idx],   dtype=torch.long)

    train_ds = TensorDataset(train_emb, train_lbl, train_pid)
    val_ds   = TensorDataset(val_emb,   val_lbl,   val_pid)

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
        sampler      = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, counts_list


def train_lofo_cv(
    all_embeddings, all_labels, all_file_idx, all_file_names,
    model_fn, num_epochs=75, device='cuda', patience=15,
    batch_size=64, num_classes=3,
    class_names={0: "barrier", 1: "cation", 2: "anion"},
    useWeightedSampler=False, weighted_ce_power=1.0, warmup_epochs=5,
    optimizer_lr=1e-3, optimizer_weight_decay=1e-2, schedular_patience=10,
    checkpoint_dir="lofo_cv",
):
    """
    Leave-One-File-Out Cross-Validation.
    Each fold holds out one entire claudin paralog file.
    Returns per-fold results and the history of the last fold.
    """
    logo         = LeaveOneGroupOut()
    label_array  = all_labels.numpy()
    fold_results = []
    last_history = None

    useWeightedCE = True if weighted_ce_power > 0 else False

    # Ensure checkpoint dir exists
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    unique_files = np.unique(all_file_idx)
    print("=" * 62)
    print(f"  LOFO-CV  —  {len(unique_files)} folds")
    print(f"  Classes: {class_names}")
    print("=" * 62)

    for fold_idx, (train_idx, val_idx) in enumerate(
        logo.split(np.arange(len(label_array)), label_array, groups=all_file_idx)
    ):
        held_out_file  = np.unique(all_file_names[val_idx])[0]
        held_out_class = int(np.unique(label_array[val_idx])[0])
        class_name     = class_names[held_out_class]

        print("=" * 62)
        print(f"Fold {fold_idx+1}/{len(unique_files)}")
        print(f"  Held-out file : {held_out_file}  (class: {class_name})")
        print(f"  Train: {len(train_idx)} seqs | Val: {len(val_idx)} seqs")
        print(f"  Train class dist: {Counter(label_array[train_idx])}")
        print("=" * 62)

        # Build loaders, criterion, and fresh model
        train_loader, val_loader, counts_list = build_loaders(
            train_idx, val_idx, all_embeddings, all_labels, all_file_idx,
            batch_size=batch_size, num_classes=num_classes,
            useWeightedSampler=useWeightedSampler,
        )
        if useWeightedCE:
            criterion = make_weighted_ce(counts_list, device, power=weighted_ce_power)
        else:
            criterion = None
        model     = model_fn().to(device)

        # Train this fold
        history = train_classifier(
            model, train_loader, val_loader,
            num_epochs          = num_epochs,
            num_classes         = num_classes,
            device              = device,
            patience            = patience,
            criterion           = criterion,
            held_out_class      = held_out_class,
            held_out_class_name = class_name,
            held_out_file_name  = held_out_file,
            class_names         = class_names,
            checkpoint_path     = f"{checkpoint_dir}/best_lofo_model.pt",
            warmup_epochs           = warmup_epochs,
            optimizer_lr            = optimizer_lr,
            optimizer_weight_decay  = optimizer_weight_decay,
            schedular_patience      = schedular_patience
        )
        last_history = history

        # Record fold results
        valid_recalls = [r for r in history['val_recall'] if not np.isnan(r)]
        best_recall   = max(valid_recalls) if valid_recalls else float('nan')

        valid_conf = [c for c in history['val_confidence'] if not np.isnan(c)]
        best_conf  = max(valid_conf) if valid_conf else float('nan')

        best_epoch_idx = history['saved_epoch'] - 1
        y_true_best = np.asarray(history['val_targets'][best_epoch_idx])
        y_prob_best = np.asarray(history['val_probs'][best_epoch_idx])
        y_pred_best = y_prob_best.argmax(axis=1)

        best_epoch_recall = history['val_recall'][best_epoch_idx]
        best_epoch_conf   = history['val_confidence'][best_epoch_idx]

        fold_results.append({
            "fold":               fold_idx + 1,
            "held_out_file":      held_out_file,
            "held_out_class":     class_name,
            "held_out_class_idx": held_out_class,
            "best_val_acc":       max(history["val_acc"]),
            "best_recall":        best_recall,
            "best_confidence":    best_conf,
            "saved_epoch":        history['saved_epoch'],
            "best_epoch_recall":  best_epoch_recall,
            "best_epoch_conf":    best_epoch_conf,
            "history":            history,
            "y_true":             y_true_best,
            "y_pred":             y_pred_best,
            "y_prob":             y_prob_best,
        })

    # LOFO summary 
    print("\n" + "=" * 90)
    print("LOFO-CV SUMMARY — Honest Per-Class Generalisation")
    print("=" * 90)
    print(f"  {'Fold':<5} {'File':<28} {'Class':<10} "
          f"{'Recall':>8} {'Conf%':<8} {'Acc%':<8} "
          f"{'Best epoch':<10} {'Best epoch Recall%':>15} {'Best epoch Conf%':>15} ")
    print(f"  {'-'*90}")

    for r in fold_results:
        recall_str = f"{r['best_recall']:.1f}%" if not np.isnan(r['best_recall']) else "  n/a "
        conf_str   = f"{r['best_confidence']:.1f}%" if not np.isnan(r['best_confidence']) else "  n/a "
        best_epoch_recall_str = f"{r['best_epoch_recall']:.1f}%" if not np.isnan(r['best_epoch_recall']) else "  n/a "
        best_epoch_con_str    = f"{r['best_epoch_conf']:.1f}%" if not np.isnan(r['best_epoch_conf']) else "  n/a "
        print(f"  {r['fold']:<5} {str(r['held_out_file']):<28} "
              f"{r['held_out_class']:<10} "
              f"{recall_str:>8} {conf_str:<8} {r['best_val_acc']:<8.1f} "
              f"{r['saved_epoch']:<10} {best_epoch_recall_str:>15} {best_epoch_con_str:>15} ")

    print(f"\n  Per-class mean recall:")
    print(f"  {'-'*40}")
    for cname in class_names.values():
        class_folds = [r for r in fold_results if r['held_out_class'] == cname]
        recalls     = [r['best_recall'] for r in class_folds if not np.isnan(r['best_recall'])]
        mean_recall = np.mean(recalls) if recalls else float('nan')
        n_folds     = len(class_folds)
        bar_val     = mean_recall if not np.isnan(mean_recall) else 0.0
        filled      = int(bar_val // 10)
        bar         = "█" * filled + "░" * (10 - filled)
        print(f"  {cname:<10} [{bar}]  {mean_recall:.1f}%  ({n_folds} fold(s))")

    mean_acc = np.mean([r["best_val_acc"] for r in fold_results])
    print(f"\n  Mean Val Acc : {mean_acc:.1f}%")
    print(f"\n  NOTE: Recall = did the model correctly identify the held-out")
    print(f"  class from a paralog it has never seen? This is the honest")
    print(f"  measure of functional generalisation.")
    print("=" * 62)

    return fold_results, last_history

# 1. Fold-wise LOFO summary table

def print_lofo_fold_summary(fold_results, save_path=None):
    rows = []
    for r in fold_results:
        rows.append({
            'Fold': r['fold'],
            'Held-out file': r['held_out_file'],
            'Held-out class': r['held_out_class'],
            'Saved epoch': r.get('saved_epoch', np.nan),
            'Saved epoch Recall (%)': r['best_epoch_recall'],
            'Saved epoch Conf (%)': r['best_epoch_conf'],
            'Best Val Acc (%)': r['best_val_acc'],
            'Best Recall (%)': r['best_recall'],
            'Best Confidence (%)': r['best_confidence'],
        })

    df = pd.DataFrame(rows)
    numeric_cols = ['Saved epoch Recall (%)', 'Saved epoch Conf (%)', 'Best Val Acc (%)', 'Best Recall (%)', 'Best Confidence (%)']
    mean_row = {'Fold': 'Mean', 'Held-out file': '-', 'Held-out class': '-', 'Saved epoch': '-'}
    std_row  = {'Fold': 'Std',  'Held-out file': '-', 'Held-out class': '-', 'Saved epoch': '-'}
    for c in numeric_cols:
        mean_row[c] = df[c].mean()
        std_row[c]  = df[c].std(ddof=1)

    df_full = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    print("\nLOFO FOLD-WISE SUMMARY")
    print(tabulate(df_full.round(2), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        df_full.to_csv(f"{save_path}/lofo_fold_summary.csv", index=False)
    return df_full


# 2. Per-class recall summary across LOFO folds

def print_lofo_class_recall_summary(fold_results, class_names=['barrier', 'cation', 'anion'], save_path=None):
    rows = []
    for cname in class_names:
        class_folds = [r for r in fold_results if r['held_out_class'] == cname]
        recalls = [r['best_recall'] for r in class_folds if not np.isnan(r['best_recall'])]
        confs   = [r['best_confidence'] for r in class_folds if not np.isnan(r['best_confidence'])]
        rows.append({
            'Class': cname,
            'Num folds': len(class_folds),
            'Mean Recall (%)': np.mean(recalls) if recalls else np.nan,
            'Std Recall (%)': np.std(recalls, ddof=1) if len(recalls) > 1 else 0.0,
            'Mean Confidence (%)': np.mean(confs) if confs else np.nan,
            'Std Confidence (%)': np.std(confs, ddof=1) if len(confs) > 1 else 0.0,
        })

    df = pd.DataFrame(rows)
    print("\nLOFO PER-CLASS GENERALISATION SUMMARY")
    print(tabulate(df.round(2), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        df.to_csv(f"{save_path}/lofo_class_recall_summary.csv", index=False)
    return df


# 3. Aggregated out-of-fold multiclass summary

def summarize_aggregated_oof_metrics(fold_results, class_names=['barrier', 'cation', 'anion'], save_path=None):
    y_true_all = np.concatenate([np.asarray(fr['y_true']) for fr in fold_results])
    y_pred_all = np.concatenate([np.asarray(fr['y_pred']) for fr in fold_results])
    y_prob_all = np.concatenate([np.asarray(fr['y_prob']) for fr in fold_results], axis=0)

    acc         = 100 * accuracy_score(y_true_all, y_pred_all)
    bal_acc     = 100 * balanced_accuracy_score(y_true_all, y_pred_all)
    macro_p     = 100 * precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)[0]
    macro_r     = 100 * precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)[1]
    macro_f1    = 100 * f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    weighted_f1 = 100 * f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)

    try:
        macro_auc = 100 * roc_auc_score(y_true_all, y_prob_all, multi_class='ovr', average='macro')
    except Exception:
        macro_auc = np.nan

    df = pd.DataFrame([{
        'Accuracy (%)': acc, 'Balanced Accuracy (%)': bal_acc,
        'Macro Precision (%)': macro_p, 'Macro Recall (%)': macro_r,
        'Macro F1 (%)': macro_f1, 'Weighted F1 (%)': weighted_f1,
        'Macro AUC OvR (%)': macro_auc, 'Total OOF Samples': len(y_true_all),
    }])

    print("\nAGGREGATED OUT-OF-FOLD MULTICLASS PERFORMANCE")
    print(tabulate(df.round(2), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        df.to_csv(f"{save_path}/lofo_aggregated_metrics.csv", index=False)
    return df


# 4. Aggregated out-of-fold per-class table

def aggregated_oof_classification_table(fold_results, class_names=['barrier', 'cation', 'anion'], save_path=None):
    y_true_all = np.concatenate([np.asarray(fr['y_true']) for fr in fold_results])
    y_pred_all = np.concatenate([np.asarray(fr['y_pred']) for fr in fold_results])
    y_prob_all = np.concatenate([np.asarray(fr['y_prob']) for fr in fold_results], axis=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=np.arange(len(class_names)), zero_division=0,
    )

    rows = []
    for i, cname in enumerate(class_names):
        try:
            auc_i = 100 * roc_auc_score((y_true_all == i).astype(int), y_prob_all[:, i])
        except Exception:
            auc_i = np.nan
        rows.append({
            'Class': cname,
            'Precision (%)': 100 * precision[i], 'Recall (%)': 100 * recall[i],
            'F1-score (%)': 100 * f1[i], 'Support': int(support[i]),
            'AUC OvR (%)': auc_i,
        })

    df = pd.DataFrame(rows)
    macro_row = pd.DataFrame([{
        'Class': 'Macro Avg',
        'Precision (%)': 100 * np.mean(precision), 'Recall (%)': 100 * np.mean(recall),
        'F1-score (%)': 100 * np.mean(f1), 'Support': int(np.sum(support)),
        'AUC OvR (%)': np.nanmean(df['AUC OvR (%)'].values),
    }])
    df_full = pd.concat([df, macro_row], ignore_index=True)

    print("\nAGGREGATED OUT-OF-FOLD PER-CLASS METRICS")
    print(tabulate(df_full.round(2), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        df_full.to_csv(f"{save_path}/lofo_classification_table.csv", index=False)
    return df_full


# 5. Aggregated confusion matrix

def plot_oof_confusion_matrix(
    fold_results, class_names=['barrier', 'cation', 'anion'],
    normalize=True, figsize=(6, 5), cmap='Blues', save_path=None,
):
    y_true_all = np.concatenate([np.asarray(fr['y_true']) for fr in fold_results])
    y_pred_all = np.concatenate([np.asarray(fr['y_pred']) for fr in fold_results])

    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(len(class_names)))
    if normalize:
        cm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        cm_display = 100 * cm
        fmt, title = '.1f', 'Aggregated Out-of-Fold Confusion Matrix (%)'
    else:
        cm_display = cm
        fmt, title = 'd', 'Aggregated Out-of-Fold Confusion Matrix (Counts)'

    plt.figure(figsize=figsize)
    sns.heatmap(cm_display, annot=np.round(cm_display, 1) if normalize else cm,
                fmt=fmt, cmap=cmap, xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/lofo_confusion_matrix.png", bbox_inches='tight', dpi=300)
    plt.show()


# 6. Aggregated ROC curves

def plot_oof_roc_curves(fold_results, class_names=['barrier', 'cation', 'anion'], figsize=None, save_path=None):
    y_true_all = np.concatenate([np.asarray(fr['y_true']) for fr in fold_results])
    y_prob_all = np.concatenate([np.asarray(fr['y_prob']) for fr in fold_results], axis=0)

    n_classes = len(class_names)
    figsize   = figsize or (5 * n_classes, 5)
    fig, axes = plt.subplots(1, n_classes, figsize=figsize, sharex=True, sharey=True)
    if n_classes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        y_true_bin = (y_true_all == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob_all[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        ax.set_title(f"OOF ROC: '{class_names[i]}'")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

    fig.suptitle('Aggregated Out-of-Fold ROC Curves', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/lofo_roc_curves.png", bbox_inches='tight', dpi=300)
    plt.show()


# 7. Per-fold val_recall and confidence curves

def plot_lofo_training_curves(fold_results, save_path=None):
    palette = plt.cm.tab10.colors

    fig, axes = plt.subplots(3, 1, figsize=(15, 14))

    for i, fr in enumerate(fold_results):
        history = fr['history']
        color   = palette[i % len(palette)]
        label   = f"{fr['held_out_file']} ({fr['held_out_class']})"

        # val_recall
        recall_vals = np.asarray(history['val_recall'], dtype=np.float32)
        epochs_r    = np.arange(1, len(recall_vals) + 1)
        axes[0].plot(epochs_r, recall_vals, color=color, linewidth=2,
                     label=label, marker='o', markersize=3, alpha=0.6)

        # val_confidence
        conf_vals = np.asarray(history['val_confidence'], dtype=np.float32)
        epochs_c  = np.arange(1, len(conf_vals) + 1)
        axes[1].plot(epochs_c, conf_vals, color=color, linewidth=2,
                     label=label, marker='o', markersize=3, alpha=0.6)

        # learning rate
        lr_vals  = np.asarray(history['lr'], dtype=np.float64)
        epochs_l = np.arange(1, len(lr_vals) + 1)
        axes[2].plot(epochs_l, lr_vals, color=color, linewidth=2,
                     label=label, marker='o', markersize=3, alpha=0.6)

    axes[0].set(
        title='Val Recall per Fold Across Training',
        xlabel='Epoch', ylabel='Held-out Class Recall (%)',
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc='upper right')

    axes[1].set(
        title='Val Confidence per Fold Across Training',
        xlabel='Epoch', ylabel='Held-out Class Confidence (%)',
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc='upper right')

    axes[2].set(
        title='Learning Rate per Fold Across Training',
        xlabel='Epoch', ylabel='Learning Rate',
    )
    axes[2].set_yscale('log')
    axes[2].yaxis.set_major_formatter(
        plt.matplotlib.ticker.LogFormatterSciNotation()
    )
    axes[2].grid(True, alpha=0.3, which='both')
    axes[2].legend(fontsize=7, loc='upper right')

    fig.suptitle('Per-Fold Training Dynamics', fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/lofo_training_curves.png", bbox_inches='tight', dpi=300)
    plt.show()


# 8. Mean LOFO primary metric curves

def plot_lofo_primary_metric_curve(
    fold_results, metric='val_recall',
    ylabel='Held-out class recall (%)',
    title='Mean Held-out-Class Recall Across LOFO Folds',
    save_path=None,
):
    histories = [fr['history'] for fr in fold_results]
    valid_histories = [
        np.asarray(h.get(metric, []), dtype=np.float32)
        for h in histories if len(h.get(metric, [])) > 0
    ]
    if not valid_histories:
        print(f"No histories found for metric: {metric}")
        return

    min_epochs = min(len(arr) for arr in valid_histories)
    epochs = np.arange(1, min_epochs + 1)
    values = np.array([arr[:min_epochs] for arr in valid_histories], dtype=np.float32)

    mean_vals = np.nanmean(values, axis=0)
    std_vals  = np.nanstd(values, axis=0)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, mean_vals, color='tab:purple', linewidth=2, label=metric)
    plt.fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals,
                     color='tab:purple', alpha=0.2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/lofo_primary_metric_curve.png", bbox_inches='tight', dpi=300)
    plt.show()