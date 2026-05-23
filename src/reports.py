from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
# Compact final checkpoint summary 

def final_checkpoint_summary_df(history):
    best_epoch  = history['saved_epoch']
    final_epoch = len(history['train_loss'])
    return pd.DataFrame([{
        'Saved epoch': best_epoch,
        'Final epoch': final_epoch,
        'Train loss @ saved': history['train_loss'][best_epoch - 1],
        'Train acc (%) @ saved': history['train_acc'][best_epoch - 1],
        'In-sample eval loss @ saved': history['val_loss'][best_epoch - 1],
        'In-sample eval acc (%) @ saved': history['val_acc'][best_epoch - 1],
        'Macro F1 (%) @ saved': history['val_macro_f1'][best_epoch - 1],
        'Macro AUC (%) @ saved': history['val_auc'][best_epoch - 1],
    }])

# Grouped Holdout summary
def grouped_checkpoint_summary_df(grouped_results):
    rows = []
    for r in grouped_results:
        history = r["history"]
        best_epoch = history["saved_epoch"]
        final_epoch = len(history["train_loss"])

        rows.append({
            "Fold": r["fold"],
            "Test files": ",".join(r["test_files"]),
            "Distribution": r["distribution"],
            "Shuffle test MSA": r["shuffle_test_msa"],
            "Saved epoch": best_epoch,
            "Final epoch": final_epoch,
            "Train loss @ saved": history["train_loss"][best_epoch - 1],
            "Train acc (%) @ saved": history["train_acc"][best_epoch - 1],
            "Val loss @ saved": history["val_loss"][best_epoch - 1],
            "Val acc (%) @ saved": history["val_acc"][best_epoch - 1],
            "Macro F1 (%) @ saved": history["val_macro_f1"][best_epoch - 1],
            "Macro AUC @ saved": history["val_auc"][best_epoch - 1],
        })

    return pd.DataFrame(rows)

def _grouped_fold_summary_df(grouped_results):
    rows = []
    for r in grouped_results:
        rows.append({
            "Fold": r["fold"],
            "Test files": ",".join(r["test_files"]),
            "Distribution": r["distribution"],
            "Shuffle": r["shuffle_test_msa"],
            "Saved epoch": r["saved_epoch"],
            "Accuracy (%)": r["accuracy"],
            "Macro Recall (%)": r["macro_recall"],
            "Macro F1 (%)": r["macro_f1"],
            "Barrier Recall (%)": r["recall_barrier"],
            "Cation Recall (%)": r["recall_cation"],
            "Anion Recall (%)": r["recall_anion"],
            "Mean Abs Prop Error": r["mean_abs_prop_error"],
            "Balanced k": r["build_meta"]["balanced_k"],
            "N test": r["build_meta"]["n_test_total"],
        })
    return pd.DataFrame(rows)

def _grouped_oog_summary_df(grouped_results, class_names):
    """
    Aggregate all grouped holdout predictions across folds.
    OOG = out-of-group (analogous to OOF in LOFO).
    """
    pred_dfs = [r["pred_df"] for r in grouped_results]
    all_preds = pd.concat(pred_dfs, ignore_index=True)

    y_true_all = all_preds["true_label"].to_numpy()
    y_pred_all = all_preds["pred_label"].to_numpy()
    y_prob_all = all_preds[["prob_barrier", "prob_cation", "prob_anion"]].to_numpy()

    try:
        macro_auc = 100 * roc_auc_score(
            y_true_all, y_prob_all, multi_class='ovr', average='macro'
        )
    except Exception:
        macro_auc = np.nan

    prfs_macro = precision_recall_fscore_support(
        y_true_all, y_pred_all, average='macro', zero_division=0
    )

    return pd.DataFrame([{
        "Accuracy (%)": 100 * accuracy_score(y_true_all, y_pred_all),
        "Balanced Accuracy (%)": 100 * balanced_accuracy_score(y_true_all, y_pred_all),
        "Macro Precision (%)": 100 * prfs_macro[0],
        "Macro Recall (%)": 100 * prfs_macro[1],
        "Macro F1 (%)": 100 * f1_score(y_true_all, y_pred_all, average='macro', zero_division=0),
        "Weighted F1 (%)": 100 * f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0),
        "Macro AUC OvR (%)": macro_auc,
        "Total OOG Samples": len(y_true_all),
    }])

def _grouped_class_summary_df(grouped_results, class_names):
    rows = []
    metric_keys = {
        "barrier": "recall_barrier",
        "cation": "recall_cation",
        "anion": "recall_anion",
    }

    for cname in class_names:
        key = metric_keys[cname]
        vals = [r[key] for r in grouped_results if not np.isnan(r[key])]
        rows.append({
            "Class": cname,
            "Num folds": len(vals),
            "Mean Recall (%)": np.mean(vals) if len(vals) > 0 else np.nan,
            "Std Recall (%)": np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
        })

    return pd.DataFrame(rows)

def _grouped_composition_summary_df(grouped_results):
    rows = []
    for r in grouped_results:
        comp_df = r["composition_df"]

        row = {
            "Fold": r["fold"],
            "Test files": ",".join(r["test_files"]),
            "Distribution": r["distribution"],
            "Shuffle": r["shuffle_test_msa"],
            "Mean Abs Prop Error": r["mean_abs_prop_error"],
        }

        for cls in [0, 1, 2]:
            row[f"True prop class {cls}"] = comp_df.loc[comp_df["class_idx"] == cls, "true_prop"].values[0]
            row[f"Pred prop class {cls}"] = comp_df.loc[comp_df["class_idx"] == cls, "pred_prop"].values[0]
            row[f"Abs prop error class {cls}"] = comp_df.loc[comp_df["class_idx"] == cls, "abs_prop_error"].values[0]

        rows.append(row)

    return pd.DataFrame(rows)

# Final training diagnostics 

def report_final_training_diagnostics(
    history, class_names=['barrier', 'cation', 'anion'],
    show_confusion=False, save_path=None,
):
    """Print diagnostic summary for the final all-data model."""
    print("\n" + "=" * 78)
    print("FINAL MODEL TRAINING DIAGNOSTICS")
    print("=" * 78)
    print("In-sample evaluation monitors optimization and checkpoint selection only.")
    print("Unbiased generalisation performance is reported from LOFO-CV.")
    print("=" * 78)
    print("⚠️ The in-sample macro F1 and AUC are expected to be high (possibly near-perfect) since there's no held-out data — these numbers should not be interpreted as generalisation performance.")

    df = final_checkpoint_summary_df(history)
    print("\nCOMPACT CHECKPOINT SUMMARY")
    print(tabulate(df.round(3), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        df.to_csv(f"{save_path}/final_diagnostics_summary.csv", index=False)

    # Training curves
    epochs     = np.arange(1, len(history['train_loss']) + 1)
    best_epoch = history['saved_epoch']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='In-sample eval', linewidth=2)
    axes[0].axvline(best_epoch, color='gray', linestyle=':', linewidth=1.2, label='Saved epoch')
    axes[0].set(title='Loss', xlabel='Epoch', ylabel='Loss')
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], label='In-sample eval', linewidth=2)
    axes[1].axvline(best_epoch, color='gray', linestyle=':', linewidth=1.2, label='Saved epoch')
    axes[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    axes[2].plot(epochs, history['lr'], color='tab:green', linewidth=2)
    axes[2].axvline(best_epoch, color='gray', linestyle=':', linewidth=1.2)
    axes[2].set(title='Learning Rate', xlabel='Epoch', ylabel='LR')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Final All-Data Training Diagnostics', fontsize=15, y=1.04)
    fig.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/final_diagnostics_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Optional confusion matrix sanity check
    if show_confusion:
        y_true = np.asarray(history['best_val_targets'])
        y_pred = np.asarray(history['best_val_preds'])
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
        cm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        cm_display = 100 * cm

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_display, annot=np.round(cm_display, 1), fmt='.1f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, cbar=True)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Final Model In-Sample Confusion Matrix (%)')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/final_diagnostics_confusion.png", dpi=300, bbox_inches='tight')
        plt.show()

    return df


# LOFO helper DataFrames

def _lofo_fold_summary_df(fold_results):
    rows = []
    for r in fold_results:
        rows.append({
            'Fold': r['fold'], 'Held-out file': r['held_out_file'],
            'Held-out class': r['held_out_class'],
            'Saved epoch': r.get('saved_epoch', np.nan),
            'Saved epoch recall (%)': r['best_epoch_recall'],
            'Saved epoch conf (%)': r['best_epoch_conf'],
            'Best Val Acc (%)': r['best_val_acc'],
            'Best Recall (%)': r['best_recall'],
            'Best Confidence (%)': r['best_confidence'],
        })
    return pd.DataFrame(rows)


def _lofo_class_summary_df(fold_results, class_names):
    rows = []
    for cname in class_names:
        class_folds = [r for r in fold_results if r['held_out_class'] == cname]
        recalls = [r['best_recall'] for r in class_folds if not np.isnan(r['best_recall'])]
        confs = [r['best_confidence'] for r in class_folds if not np.isnan(r['best_confidence'])]
        rows.append({
            'Class': cname, 'Num folds': len(class_folds),
            'Mean Recall (%)': np.mean(recalls) if recalls else np.nan,
            'Std Recall (%)': np.std(recalls, ddof=1) if len(recalls) > 1 else 0.0,
            'Mean Confidence (%)': np.mean(confs) if confs else np.nan,
            'Std Confidence (%)': np.std(confs, ddof=1) if len(confs) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def _lofo_oof_summary_df(fold_results, class_names):
    y_true_all = np.concatenate([np.asarray(fr['y_true']) for fr in fold_results])
    y_pred_all = np.concatenate([np.asarray(fr['y_pred']) for fr in fold_results])
    y_prob_all = np.concatenate([np.asarray(fr['y_prob']) for fr in fold_results], axis=0)

    try:
        macro_auc = 100 * roc_auc_score(y_true_all, y_prob_all, multi_class='ovr', average='macro')
    except Exception:
        macro_auc = np.nan

    return pd.DataFrame([{
        'Accuracy (%)': 100 * accuracy_score(y_true_all, y_pred_all),
        'Balanced Accuracy (%)': 100 * balanced_accuracy_score(y_true_all, y_pred_all),
        'Macro Precision (%)': 100 * precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)[0],
        'Macro Recall (%)': 100 * precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)[1],
        'Macro F1 (%)': 100 * f1_score(y_true_all, y_pred_all, average='macro', zero_division=0),
        'Weighted F1 (%)': 100 * f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0),
        'Macro AUC OvR (%)': macro_auc,
        'Total OOF Samples': len(y_true_all),
    }])


# LOFO-CV main scientific report
def report_lofo_cv_main(
    fold_results, class_names=['barrier', 'cation', 'anion'],
    show_training_curves=True, show_roc=True, save_path=None,
):
    """Generate the full LOFO-CV report with tables, confusion matrix, ROC curves, and training dynamics."""
    print("\n" + "=" * 78)
    print("LOFO-CV MAIN EVALUATION REPORT")
    print("=" * 78)
    print("Unbiased performance under leave-one-file-out cross-validation.")
    print("Each sample contributes only via out-of-fold predictions from a held-out paralog.")
    print("=" * 78)

    # 1. Fold-level table
    fold_df = _lofo_fold_summary_df(fold_results)
    print("\n1) FOLD-WISE HELD-OUT GENERALISATION")
    print(tabulate(fold_df.round(2), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        fold_df.to_csv(f"{save_path}/lofo_fold_summary.csv", index=False)

    # 2. Per-class table
    class_df = _lofo_class_summary_df(fold_results, class_names)
    print("\n2) PER-CLASS HELD-OUT GENERALISATION")
    print(tabulate(class_df.round(2), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        class_df.to_csv(f"{save_path}/lofo_class_summary.csv", index=False)

    # 3. Aggregated OOF summary
    oof_df = _lofo_oof_summary_df(fold_results, class_names)
    print("\n3) AGGREGATED OUT-OF-FOLD MULTICLASS PERFORMANCE")
    print(tabulate(oof_df.round(2), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        oof_df.to_csv(f"{save_path}/lofo_oof_summary.csv", index=False)

    # 4. OOF confusion matrix
    y_true_all = np.concatenate([np.asarray(fr['y_true']) for fr in fold_results])
    y_pred_all = np.concatenate([np.asarray(fr['y_pred']) for fr in fold_results])
    y_prob_all = np.concatenate([np.asarray(fr['y_prob']) for fr in fold_results], axis=0)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(len(class_names)))
    cm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    cm_display = 100 * cm

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_display, annot=np.round(cm_display, 1), fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Aggregated Out-of-Fold Confusion Matrix (%)')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/lofo_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 5. OOF ROC curves
    if show_roc:
        n_classes = len(class_names)
        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5), sharex=True, sharey=True)
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
            plt.savefig(f"{save_path}/lofo_roc_curves.png", dpi=300, bbox_inches='tight')
        plt.show()

    # 6. Per-fold val_recall and confidence curves
    if show_training_curves:
        palette = plt.cm.tab10.colors

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

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

        fig.suptitle('Per-Fold Training Dynamics', fontsize=14, y=1.02)
        fig.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/lofo_training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()

    return {
        'fold_summary': fold_df,
        'class_summary': class_df,
        'oof_summary': oof_df,
    }

# Grouped holdout main scientific report
def report_grouped_holdout_main(
    grouped_results,
    class_names=['barrier', 'cation', 'anion'],
    show_training_curves=True,
    show_roc=False,
    show_composition=True,
    save_path=None,
):
    """
    Scientific report for grouped mixed-MSA holdout evaluation.
    """
    print("\n" + "=" * 78)
    print("GROUPED MIXED HOLDOUT MAIN EVALUATION REPORT")
    print("=" * 78)

    if len(grouped_results) == 0:
        print("No grouped results found.")
        return None

    distribution = grouped_results[0]["distribution"]
    shuffle_test_msa = grouped_results[0]["shuffle_test_msa"]

    print("Unbiased held-out evaluation using grouped mixed-MSA test folds.")
    print("Each fold holds out 1 barrier + 1 cation + 1 anion source MSA,")
    print("then recomputes embeddings jointly in mixed-MSA context.")
    print("=" * 78)
    print(f"Distribution mode : {distribution}")
    print(f"Shuffle test MSA  : {shuffle_test_msa}")

    # 1. Fold-level table
    fold_df = _grouped_fold_summary_df(grouped_results)
    print("\n1) FOLD-WISE GROUPED GENERALISATION")
    print(tabulate(fold_df.round(3), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        fold_df.to_csv(f"{save_path}/grouped_fold_summary.csv", index=False)

    # 2. Per-class summary
    class_df = _grouped_class_summary_df(grouped_results, class_names)
    print("\n2) PER-CLASS GROUPED GENERALISATION")
    print(tabulate(class_df.round(3), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        class_df.to_csv(f"{save_path}/grouped_class_summary.csv", index=False)

    # 3. Aggregated OOG summary
    oog_df = _grouped_oog_summary_df(grouped_results, class_names)
    print("\n3) AGGREGATED OUT-OF-GROUP MULTICLASS PERFORMANCE")
    print(tabulate(oog_df.round(3), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        oog_df.to_csv(f"{save_path}/grouped_oog_summary.csv", index=False)

    # 4. Checkpoint summary
    ckpt_df = grouped_checkpoint_summary_df(grouped_results)
    print("\n4) COMPACT CHECKPOINT SUMMARY")
    print(tabulate(ckpt_df.round(3), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))
    if save_path:
        ckpt_df.to_csv(f"{save_path}/grouped_checkpoint_summary.csv", index=False)

    # 5. Composition summary
    comp_df = None
    if show_composition:
        comp_df = _grouped_composition_summary_df(grouped_results)
        print("\n5) COMPOSITION RECOVERY SUMMARY")
        print(tabulate(comp_df.round(4), headers='keys', tablefmt='fancy_grid',
                       showindex=False, numalign='center'))
        if save_path:
            comp_df.to_csv(f"{save_path}/grouped_composition_summary.csv", index=False)

    # 6. Aggregated confusion matrix
    pred_dfs = [r["pred_df"] for r in grouped_results]
    all_preds = pd.concat(pred_dfs, ignore_index=True)

    y_true_all = all_preds["true_label"].to_numpy()
    y_pred_all = all_preds["pred_label"].to_numpy()
    y_prob_all = all_preds[["prob_barrier", "prob_cation", "prob_anion"]].to_numpy()

    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(len(class_names)))
    cm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    cm_display = 100 * cm

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_display, annot=np.round(cm_display, 1), fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Aggregated Grouped-Holdout Confusion Matrix (%)')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/grouped_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 7. ROC curves
    if show_roc:
        n_classes = len(class_names)
        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5), sharex=True, sharey=True)
        if n_classes == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            y_true_bin = (y_true_all == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, y_prob_all[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
            ax.set_title(f"Grouped ROC: '{class_names[i]}'")
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')

        fig.suptitle('Aggregated Grouped-Holdout ROC Curves', fontsize=14, y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/grouped_roc_curves.png", dpi=300, bbox_inches='tight')
        plt.show()

    # 8. Per-fold training curves
    if show_training_curves:
        palette = plt.cm.tab20.colors

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        for i, r in enumerate(grouped_results):
            history = r["history"]
            color = palette[i % len(palette)]
            label = f"Fold {r['fold']}"

            epochs = np.arange(1, len(history["train_loss"]) + 1)

            axes[0, 0].plot(epochs, history["train_loss"], color=color, linewidth=1.8, alpha=0.8, label=label)
            axes[0, 1].plot(epochs, history["val_loss"], color=color, linewidth=1.8, alpha=0.8, label=label)
            axes[1, 0].plot(epochs, history["val_acc"], color=color, linewidth=1.8, alpha=0.8, label=label)

            macro_f1_vals = np.asarray(history["val_macro_f1"], dtype=np.float32)
            axes[1, 1].plot(epochs, macro_f1_vals, color=color, linewidth=1.8, alpha=0.8, label=label)

        axes[0, 0].set(title='Train Loss per Fold', xlabel='Epoch', ylabel='Loss')
        axes[0, 1].set(title='Val Loss per Fold', xlabel='Epoch', ylabel='Loss')
        axes[1, 0].set(title='Val Accuracy per Fold', xlabel='Epoch', ylabel='Accuracy (%)')
        axes[1, 1].set(title='Val Macro F1 per Fold', xlabel='Epoch', ylabel='Macro F1')

        for ax in axes.ravel():
            ax.grid(True, alpha=0.3)

        axes[0, 0].legend(fontsize=7, loc='upper right')
        fig.suptitle('Grouped-Holdout Training Dynamics', fontsize=14, y=1.02)
        fig.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/grouped_training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()

    return {
        "fold_summary": fold_df,
        "class_summary": class_df,
        "oog_summary": oog_df,
        "checkpoint_summary": ckpt_df,
        "composition_summary": comp_df,
    }

def report_grouped_holdout_compact(grouped_results, save_path=None):
    print("\n" + "=" * 78)
    print("GROUPED MIXED HOLDOUT COMPACT REPORT")
    print("=" * 78)

    fold_df = _grouped_fold_summary_df(grouped_results)
    ckpt_df = grouped_checkpoint_summary_df(grouped_results)

    print("\nFOLD SUMMARY")
    print(tabulate(fold_df.round(3), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))

    print("\nCHECKPOINT SUMMARY")
    print(tabulate(ckpt_df.round(3), headers='keys', tablefmt='fancy_grid',
                   showindex=False, numalign='center'))

    if save_path:
        fold_df.to_csv(f"{save_path}/grouped_compact_fold_summary.csv", index=False)
        ckpt_df.to_csv(f"{save_path}/grouped_compact_checkpoint_summary.csv", index=False)

    return {
        "fold_summary": fold_df,
        "checkpoint_summary": ckpt_df,
    }
