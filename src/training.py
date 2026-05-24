from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from sklearn.metrics import f1_score, roc_auc_score
from tabulate import tabulate

def compute_honest_val_metrics(
    val_targets,
    val_probs,
    held_out_class,
    class_names={0: "barrier", 1: "cation", 2: "anion"},
):
    """Compute metrics valid when validation has one dominant held-out class."""
    val_targets = np.array(val_targets)
    val_probs = np.array(val_probs)
    val_preds = val_probs.argmax(axis=1)
    n_classes = val_probs.shape[1]

    acc = (val_preds == val_targets).mean() * 100

    mask_true = val_targets == held_out_class
    recall = (
        (val_preds[mask_true] == held_out_class).mean() * 100
        if mask_true.any() else 0.0
    )

    wrong_preds = val_preds[val_preds != held_out_class]
    confusion = Counter(wrong_preds.tolist())
    confusion_str = ", ".join(
        f"{class_names[k]}({v})" for k, v in sorted(confusion.items())
    ) if confusion else "none"

    correct_class_conf = val_probs[:, held_out_class].mean() * 100
    mean_probs = val_probs.mean(axis=0)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
    max_entropy = np.log(n_classes)
    confidence_level = "decisive" if entropy < 0.5 * max_entropy else "uncertain"

    return {
        "acc": acc,
        "recall": recall,
        "correct_class_conf": correct_class_conf,
        "confusion_str": confusion_str,
        "entropy": entropy,
        "confidence_level": confidence_level,
        "n_val": len(val_targets),
    }


def compute_full_val_metrics(
    val_targets,
    val_probs,
    class_names={0: "barrier", 1: "cation", 2: "anion"},
):
    """Compute multi-class metrics when all classes may be present in validation."""
    val_targets = np.array(val_targets)
    val_probs = np.array(val_probs)
    val_preds = val_probs.argmax(axis=1)
    n_classes = val_probs.shape[1]
    all_labels = list(range(n_classes))

    acc = (val_preds == val_targets).mean() * 100

    macro_f1 = f1_score(
        val_targets,
        val_preds,
        average="macro",
        labels=all_labels,
        zero_division=0,
    )

    n_present = len(np.unique(val_targets))
    if n_present >= 2:
        try:
            auc_val = roc_auc_score(
                val_targets,
                val_probs,
                multi_class="ovr",
                average="macro",
                labels=all_labels,
            )
        except ValueError:
            auc_val = float("nan")
    else:
        auc_val = float("nan")

    per_class_recall = {}
    for cid, cname in class_names.items():
        mask = val_targets == cid
        if mask.any():
            per_class_recall[cname] = (val_preds[mask] == cid).mean() * 100
        else:
            per_class_recall[cname] = float("nan")

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "auc": auc_val,
        "per_class_recall": per_class_recall,
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
    """Print concise epoch summary for LOFO and full-validation modes."""
    is_lofo = held_out_class is not None

    val_targets = np.array(val_all_targets)
    val_probs = np.array(val_all_probs)
    val_preds = val_probs.argmax(axis=1)
    val_acc = (val_preds == val_targets).mean() * 100

    mode_label = (
        f"LOFO - held out: {held_out_file_name}"
        if is_lofo
        else "Final Training"
    )

    print(f"\n{'-'*62}")
    print(f"  Epoch {epoch}/{num_epochs}  |  {mode_label}")
    print(f"{'-'*62}")
    print(f"  [Train]  Loss: {train_loss:.4f}  |  Acc: {train_acc:.1f}%")
    print(f"  [Val]    Loss: {val_loss:.4f}  |  Acc: {val_acc:.1f}%")

    if is_lofo:
        m = compute_honest_val_metrics(
            val_targets,
            val_probs,
            held_out_class,
            class_names,
        )
        primary_metric = m["recall"]
    else:
        m = compute_full_val_metrics(val_targets, val_probs, class_names)
        primary_metric = m["macro_f1"]

    metric_label = "recall" if is_lofo else "macro F1"
    metric_fmt = f"{primary_metric:.1f}%" if is_lofo else f"{primary_metric:.4f}"
    if improved:
        print(f"  Best {metric_label} improved -> {metric_fmt}; checkpoint saved")
    else:
        print(f"  No improvement. Patience: {patience_counter}/{patience}")

    return primary_metric


def make_weighted_ce(class_counts, device=None, power=1.0):
    """Build a weighted CrossEntropyLoss inversely proportional to class frequency."""
    counts  = torch.tensor(class_counts, dtype=torch.float32)
    weights = (counts.sum() / (len(counts) * counts)) ** power
    weights = weights / weights.sum() * len(counts)
    if device is not None:
        weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

def train_classifier(
    classifier,
    train_loader,
    val_loader,
    num_epochs=75,
    num_classes=3,
    device='cuda',
    patience=15,
    criterion=None,
    checkpoint_path="model_best_epoch.pt",
    held_out_class=None,
    held_out_class_name=None,
    held_out_file_name=None,
    class_names={0: "barrier", 1: "cation", 2: "anion"},
    warmup_epochs=5,
    optimizer_lr=1e-3,
    optimizer_weight_decay=1e-2,
    schedular_patience=10
):
    """
    Unified training loop for both LOFO folds and final training.

    Modes:
      - LOFO mode  (held_out_class is int)  : tracks recall as primary metric
      - Final mode (held_out_class is None) : tracks macro F1 as primary metric
    """
    is_lofo = held_out_class is not None

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay # 3e-3 1e-4
    )

    # Phase 1: Warmup only (no cosine decay after)
    def warmup_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0          # ← hold at full LR after warmup, let ReduceLROnPlateau take over

    warmup_scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_lambda)

    # Phase 2: ReduceLROnPlateau takes over after warmup
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',         # maximising recall / F1
        patience=schedular_patience, # 2
        min_lr=1e-6,        # never let LR hit zero
    )

    history = {
        "train_loss":       [],
        "train_acc":        [],
        "val_loss":         [],
        "val_acc":          [],
        "val_recall":       [],
        "val_macro_f1":     [],
        "val_auc":          [],
        "val_confidence":   [],
        "val_targets":      [],
        "best_val_targets": [],
        "val_probs":        [],
        "best_val_probs":   [],
        "best_val_preds":   [],
        "lr":               [],
        'saved_epoch':      1,
    }

    best_primary_metric = 0.0
    patience_counter    = 0

    for epoch in range(num_epochs):
        # Training
        classifier.train()
        train_loss_sum = 0.0
        train_correct  = 0
        train_total    = 0

        for batch_x, batch_y, _ in train_loader:
            batch_x       = batch_x.to(device)
            batch_y       = batch_y.to(device)

            optimizer.zero_grad()

            class_logits = classifier(batch_x)

            L_cls = criterion(class_logits, batch_y)

            loss = L_cls
            loss.backward()

            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.size(0)
            preds           = class_logits.argmax(dim=1)
            train_correct  += (preds == batch_y).sum().item()
            train_total    += batch_x.size(0)

        train_loss = train_loss_sum / train_total
        train_acc  = train_correct / train_total * 100

        # Validation
        classifier.eval()
        val_loss_sum    = 0.0
        val_all_targets = []
        val_all_probs   = []

        with torch.no_grad():
            for batch_x, batch_y, *_ in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                class_logits = classifier(batch_x)

                v_loss = criterion(class_logits, batch_y)
                val_loss_sum += v_loss.item() * batch_x.size(0)

                probs = torch.softmax(class_logits, dim=1).cpu().numpy()
                val_all_probs.extend(probs.tolist())
                val_all_targets.extend(batch_y.cpu().numpy().tolist())

        val_loss        = val_loss_sum / len(val_all_targets)
        val_all_targets = np.array(val_all_targets)
        val_all_probs   = np.array(val_all_probs)
        val_preds       = val_all_probs.argmax(axis=1)

        # Compute primary metric for early stopping
        if is_lofo:
            m = compute_honest_val_metrics(
                val_all_targets, val_all_probs, held_out_class, class_names,
            )
            primary_metric = m['recall']
            history['val_recall'].append(m['recall'])
            history['val_confidence'].append(m['correct_class_conf'])
            history['val_macro_f1'].append(float('nan'))
            history['val_auc'].append(float('nan'))
        else:
            m = compute_full_val_metrics(
                val_all_targets, val_all_probs, class_names,
            )
            primary_metric = m['macro_f1']
            history['val_macro_f1'].append(m['macro_f1'])
            history['val_auc'].append(m['auc'])
            history['val_recall'].append(float('nan'))
            history['val_confidence'].append(float('nan'))

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append((val_preds == val_all_targets).mean() * 100)
        history['val_targets'].append(val_all_targets)
        history['val_probs'].append(val_all_probs)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Early stopping / checkpoint
        # Guard: do not save/evaluate during warmup
        in_warmup = epoch < warmup_epochs

        improved = (not in_warmup) and (primary_metric > best_primary_metric)
        if improved and checkpoint_path is not None:
            best_primary_metric = primary_metric
            patience_counter    = 0
            torch.save({
                "model_state":    classifier.state_dict(),
                "epoch":          epoch + 1,
                "primary_metric": primary_metric,
                "metric_name":    "recall" if is_lofo else "macro_f1",
                "held_out_class": held_out_class_name,
            }, checkpoint_path)
            history['saved_epoch'] = epoch + 1
        elif not in_warmup:
            patience_counter += 1

        # Epoch summary
        print_epoch_summary(
            epoch=epoch + 1,
            num_epochs=num_epochs,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_all_targets=val_all_targets,
            val_all_probs=val_all_probs,
            patience_counter=patience_counter,
            patience=patience,
            improved=improved,
            held_out_class=held_out_class,
            held_out_class_name=held_out_class_name,
            held_out_file_name=held_out_file_name,
            class_names=class_names,
        )

        # Scheduler stepping
        if in_warmup:
            warmup_scheduler.step()                   # ramp LR up during warmup
        else:
            plateau_scheduler.step(primary_metric)


        if (not in_warmup) and (patience_counter >= patience):
            print(f"\n⚠️  Early stopping at epoch {epoch+1}. "
                  f"Best {'recall' if is_lofo else 'macro F1'}: "
                  f"{best_primary_metric:.4f}")
            break

    # Record best-epoch predictions
    best_idx = history['saved_epoch'] - 1
    history['best_val_targets'] = history['val_targets'][best_idx]
    history['best_val_probs']   = history['val_probs'][best_idx]
    history['best_val_preds']   = np.argmax(history['best_val_probs'], axis=1)
    return history

def _eval_label(insample=True):
    return 'InSample' if insample else 'Validation'

# 1. Final training summary table

def print_final_training_summary(history, class_names=['barrier', 'cation', 'anion'], save_path=None, insample=True):
    eval_label = _eval_label(insample)
    best_epoch  = history['saved_epoch']
    final_epoch = len(history['train_loss'])

    row_best = {
        'Checkpoint': 'Best saved epoch', 'Epoch': best_epoch,
        'Train Loss': history['train_loss'][best_epoch - 1],
        'Train Acc (%)': history['train_acc'][best_epoch - 1],
        f'{eval_label} Loss': history['val_loss'][best_epoch - 1],
        f'{eval_label} Acc (%)': history['val_acc'][best_epoch - 1],
        'Macro F1 (%)': history['val_macro_f1'][best_epoch - 1]
            if not np.isnan(history['val_macro_f1'][best_epoch - 1]) else np.nan,
        'Macro AUC (%)': history['val_auc'][best_epoch - 1]
            if not np.isnan(history['val_auc'][best_epoch - 1]) else np.nan,
    }
    row_final = {
        'Checkpoint': 'Final epoch', 'Epoch': final_epoch,
        'Train Loss': history['train_loss'][-1],
        'Train Acc (%)': history['train_acc'][-1],
        f'{eval_label} Loss': history['val_loss'][-1],
        f'{eval_label} Acc (%)': history['val_acc'][-1],
        'Macro F1 (%)': history['val_macro_f1'][-1]
            if not np.isnan(history['val_macro_f1'][-1]) else np.nan,
        'Macro AUC (%)': history['val_auc'][-1]
            if not np.isnan(history['val_auc'][-1]) else np.nan,
    }

    df = pd.DataFrame([row_best, row_final])
    print("\nFINAL TRAINING SUMMARY")
    print(f"(Diagnostic / {eval_label.lower()} monitoring only)")
    print(tabulate(df.round(3), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(f"{save_path}/final_training_summary.csv", index=False)
    return df


# 2. Final evaluation overall metrics (optional)

def summarize_final_in_sample_metrics(history, class_names=['barrier', 'cation', 'anion'], save_path=None, insample=True):
    eval_label = _eval_label(insample)
    y_true = np.asarray(history['best_val_targets'])
    y_pred = np.asarray(history['best_val_preds'])
    y_prob = np.asarray(history['best_val_probs'])

    acc         = 100 * accuracy_score(y_true, y_pred)
    bal_acc     = 100 * balanced_accuracy_score(y_true, y_pred)
    macro_p     = 100 * precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[0]
    macro_r     = 100 * precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[1]
    macro_f1    = 100 * f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = 100 * f1_score(y_true, y_pred, average='weighted', zero_division=0)

    try:
        macro_auc = 100 * roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        macro_auc = np.nan

    df = pd.DataFrame([{
        'Saved epoch': history['saved_epoch'],
        'Accuracy (%)': acc, 'Balanced Accuracy (%)': bal_acc,
        'Macro Precision (%)': macro_p, 'Macro Recall (%)': macro_r,
        'Macro F1 (%)': macro_f1, 'Weighted F1 (%)': weighted_f1,
        'Macro AUC OvR (%)': macro_auc, 'Num samples': len(y_true),
    }])

    print(f"\nFINAL MODEL {eval_label.upper()} PERFORMANCE AT SAVED EPOCH")
    print("(Useful for optimization diagnostics; not an unbiased test estimate)")
    print(tabulate(df.round(2), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(f"{save_path}/final_{eval_label.lower()}_metrics.csv", index=False)
    return df


# 3. Final evaluation per-class table (optional)

def final_in_sample_classification_table(history, class_names=['barrier', 'cation', 'anion'], save_path=None, insample=True):
    eval_label = _eval_label(insample)
    y_true = np.asarray(history['best_val_targets'])
    y_pred = np.asarray(history['best_val_preds'])
    y_prob = np.asarray(history['best_val_probs'])

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0,
    )

    rows = []
    for i, cname in enumerate(class_names):
        try:
            auc_i = 100 * roc_auc_score((y_true == i).astype(int), y_prob[:, i])
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

    print(f"\nFINAL MODEL {eval_label.upper()} PER-CLASS METRICS")
    print("(At saved epoch; diagnostic only)")
    print(tabulate(df_full.round(2), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        df_full.to_csv(f"{save_path}/final_{eval_label.lower()}_classification_table.csv", index=False)
    return df_full


# 4. Final training curves

def plot_final_training_history(history, save_path=None, insample=True):
    eval_label = _eval_label(insample)
    epochs     = np.arange(1, len(history['train_loss']) + 1)
    best_epoch = history['saved_epoch']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label=f'{eval_label}', linewidth=2)
    axes[0].axvline(best_epoch, color='gray', linestyle=':', linewidth=1)
    axes[0].set(title='Final Training Loss', xlabel='Epoch', ylabel='Loss')
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], label=f'{eval_label}', linewidth=2)
    axes[1].axvline(best_epoch, color='gray', linestyle=':', linewidth=1)
    axes[1].set(title='Final Training Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    axes[2].plot(epochs, history['lr'], color='tab:green', linewidth=2)
    axes[2].axvline(best_epoch, color='gray', linestyle=':', linewidth=1)
    axes[2].set(title='Learning Rate Schedule', xlabel='Epoch', ylabel='Learning Rate')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Final All-Data Training History', fontsize=15, y=1.04)
    fig.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/final_training_history.png", bbox_inches='tight', dpi=300)
    plt.show()


# 5. Final macro metric curves

def plot_final_macro_metrics(history, save_path=None, insample=True):
    eval_label = _eval_label(insample)
    epochs     = np.arange(1, len(history['train_loss']) + 1)
    best_epoch = history['saved_epoch']
    macro_f1   = np.asarray(history['val_macro_f1'], dtype=np.float32)
    macro_auc  = np.asarray(history['val_auc'], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, macro_f1, color='tab:purple', linewidth=2)
    axes[0].axvline(best_epoch, color='gray', linestyle=':', linewidth=1)
    axes[0].set(title=f'{eval_label} Macro F1', xlabel='Epoch', ylabel='Macro F1 (%)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, macro_auc, color='tab:red', linewidth=2)
    axes[1].axvline(best_epoch, color='gray', linestyle=':', linewidth=1)
    axes[1].set(title=f'{eval_label} Macro AUC', xlabel='Epoch', ylabel='Macro AUC OvR (%)')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Final All-Data {eval_label} Macro Metrics', fontsize=14, y=1.03)
    fig.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/final_macro_metrics.png", bbox_inches='tight', dpi=300)
    plt.show()


# 6. Final confusion matrix

def plot_final_confusion_matrix(
    history, class_names=['barrier', 'cation', 'anion'],
    normalize=True, figsize=(6, 5), cmap='Blues', save_path=None, insample=True,
):
    eval_label = _eval_label(insample)
    y_true = np.asarray(history['best_val_targets'])
    y_pred = np.asarray(history['best_val_preds'])

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    if normalize:
        cm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        cm_display = 100 * cm
        fmt, title = '.1f', f'Final Model {eval_label} Confusion Matrix (%)'
    else:
        cm_display = cm
        fmt, title = 'd', f'Final Model {eval_label} Confusion Matrix (Counts)'

    plt.figure(figsize=figsize)
    sns.heatmap(cm_display, annot=np.round(cm_display, 1) if normalize else cm,
                fmt=fmt, cmap=cmap, xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/final_confusion_matrix.png", bbox_inches='tight', dpi=300)
    plt.show()


# 7. Final ROC curves (optional)

def plot_final_roc_curves(history, class_names=['barrier', 'cation', 'anion'], figsize=None, save_path=None, insample=True):
    eval_label = _eval_label(insample)
    y_true    = np.asarray(history['best_val_targets'])
    y_prob    = np.asarray(history['best_val_probs'])
    n_classes = len(class_names)
    figsize   = figsize or (5 * n_classes, 5)

    fig, axes = plt.subplots(1, n_classes, figsize=figsize, sharex=True, sharey=True)
    if n_classes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        y_true_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        ax.set_title(f"{eval_label} ROC: '{class_names[i]}'")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

    fig.suptitle(f'Final Model {eval_label} ROC Curves', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/final_roc_curves.png", bbox_inches='tight', dpi=300)
    plt.show()

def evaluate_split(name, model, embeddings, labels, class_names=['barrier', 'cation', 'anion']):
    model.eval()
    with torch.no_grad():
        logits = model(embeddings.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()
    y_pred = probs.argmax(axis=1)

    acc = 100 * accuracy_score(y_true, y_pred)
    bal_acc = 100 * balanced_accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\n" + "=" * 72)
    print(f"{name} RESULTS")
    print("=" * 72)
    print(f"Accuracy (%)           : {acc:.2f}")
    print(f"Balanced Accuracy (%)  : {bal_acc:.2f}")
    print(f"Macro Precision (%)    : {macro_p * 100:.2f}")
    print(f"Macro Recall (%)       : {macro_r * 100:.2f}")
    print(f"Macro F1 (%)           : {macro_f1 * 100:.2f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    print("\nPer-class metrics:")
    for i, cname in class_names.items():
        p = per_class[0][i] * 100
        r = per_class[1][i] * 100
        f1 = per_class[2][i] * 100
        print(f"  {cname:<7}  P={p:6.2f}  R={r:6.2f}  F1={f1:6.2f}")

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1 * 100,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }
