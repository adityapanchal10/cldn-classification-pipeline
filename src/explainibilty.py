from captum.attr import IntegratedGradients
from typing import Dict


def make_zero_baseline(seq_length, embed_dim=768):
    """Zero-embedding baseline for IG: represents 'no information'."""
    return torch.zeros(1, seq_length, embed_dim)


def compute_ig_attributions(model, inputs, baseline, target_class, n_steps=50, device=None):
    """Compute per-residue Integrated Gradients attributions."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device).eval()

    inputs   = inputs.to(device).requires_grad_(True)
    baseline = baseline.to(device)

    ig = IntegratedGradients(lambda inp: model(inp))
    attributions, delta = ig.attribute(
        inputs, baselines=baseline, target=target_class,
        n_steps=n_steps, return_convergence_delta=True, method='gausslegendre',
    )
    residue_attrs = torch.sum(attributions, dim=2)  # (B, seq_len)

    return residue_attrs, delta


def get_top_k_residues(residue_attributions, k=10):
    """Extract top-k most important residues (positive + negative contributors)."""
    batch_size = residue_attributions.shape[0]
    results = []
    for i in range(batch_size):
        pos_values, pos_indices = torch.topk(residue_attributions[i], k - k // 2)
        neg_values, neg_indices = torch.topk(residue_attributions[i], k // 2, largest=False)
        results.append({
            'sample_id':           i,
            'top_residue_indices': torch.cat((pos_indices, neg_indices)).cpu().tolist(),
            'top_residue_scores':  torch.cat((pos_values, neg_values)).cpu().tolist(),
            'total_attribution':   residue_attributions[i].sum().item(),
        })
    return results


def ablation_study(model, inputs, targets, residue_attrs, k_values, device=None):
    """Zero-out top-k attributed residues and measure loss increase."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device).eval()

    with torch.no_grad():
        orig_logits = model(inputs.to(device))
        orig_loss   = nn.CrossEntropyLoss()(orig_logits, targets.to(device))
        _, orig_pred = orig_logits.max(dim=1)

    results = {
        'original_loss':  orig_loss.item(),
        'original_class': orig_pred.item(),
        'ablations':      {},
    }

    for k in k_values:
        ablated = inputs.clone()
        for i in range(inputs.shape[0]):
            top_k_idx = torch.argsort(residue_attrs[i], descending=True)[:k]
            ablated[i, top_k_idx, :] = 0

        with torch.no_grad():
            abl_logits = model(ablated.to(device))
            abl_loss   = nn.CrossEntropyLoss()(abl_logits, targets.to(device))
            _, abl_pred = abl_logits.max(dim=1)

        results['ablations'][f'top_{k}'] = {
            'loss_increase':             (abl_loss - orig_loss).item(),
            'predicted_class_after_abl': abl_pred.item(),
        }

    return results


def explain_predictions(
    model, input_seq_ids, input_seqs, input_embeddings,
    baseline, predicted_classes, confidences, true_classes,
    k, n_steps, device=None, run_ablation=True,
) -> Dict:
    """Run IG explanations and optional ablation for a batch of sequences."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    batch_size = input_embeddings.shape[0]
    model.to(device).eval()

    all_results = {'samples': [], 'ablation_validation': [], 'summary': {}}

    with torch.no_grad():
        for i in range(batch_size):
            sample     = input_embeddings[i].unsqueeze(0).to(device)
            pred_class = predicted_classes[i].item()
            cf         = confidences[i].item()

            residue_attrs, delta = compute_ig_attributions(
                model=model, inputs=sample, baseline=baseline,
                target_class=pred_class, n_steps=n_steps, device=device,
            )
            top_k_results = get_top_k_residues(residue_attrs, k=k)

            sample_result = {
                'sample_id':         i,
                'input_seq_desc':    input_seq_ids[i],
                'sequence':          input_seqs[i],
                'predicted_class':   pred_class,
                'confidence':        cf,
                'attributions':      residue_attrs,
                'convergence_delta': delta[0].item(),
                **dict(list(top_k_results[0].items())[1:]),
            }
            all_results['samples'].append(sample_result)

            if run_ablation and true_classes is not None:
                true_class = true_classes[i].item()
                ablation_results = ablation_study(
                    model=model, inputs=sample,
                    targets=torch.tensor([true_class]).long(),
                    residue_attrs=residue_attrs,
                    k_values=[1, 5, 10], device=device,
                )
                ablation_results['sample_id'] = i
                all_results['ablation_validation'].append(ablation_results)

    deltas = [s['convergence_delta'] for s in all_results['samples']]
    all_results['summary'] = {
        'num_samples':             batch_size,
        'mean_convergence_delta':  sum(deltas) / len(deltas),
        'max_convergence_delta':   max(deltas),
        'ig_approximation_quality':
            'excellent' if max(deltas) < 0.05 else
            'good'      if max(deltas) < 0.1  else 'check',
    }
    return all_results


def print_results(results):
    """Pretty-print IG explanation results."""
    print("\n" + "=" * 70)
    print("INTEGRATED GRADIENTS EXPLANATION RESULTS")
    print("=" * 70)

    print("\nSummary:")
    print(f"  Samples analyzed: {results['summary']['num_samples']}")
    print(f"  Mean convergence delta: {results['summary']['mean_convergence_delta']:.6f}")
    print(f"  IG approximation quality: {results['summary']['ig_approximation_quality']}")

    for sample in results['samples']:
        sid  = sample['sample_id']
        seq  = sample['sequence']
        desc = sample['input_seq_desc'] or f"sample_{sid}"

        print(f"\n>>> Sample {sid + 1}")
        print(f"Name: {desc}")
        print(f"Sequence (len={len(seq)}): {seq}")
        print(f"Predicted Class: {sample['predicted_class']} "
              f"({CLASS_MAP[sample['predicted_class']]}), "
              f"confidence: {sample['confidence']:.3f}")
        print("\n--- Top-k Important Residues ---")

        idxs      = sample['top_residue_indices']
        scores    = sample['top_residue_scores']
        max_score = max(scores) if scores else 1.0
        n_pos     = len(idxs) - (len(idxs) // 2)

        print(" Contributing positively towards the prediction:")
        for rank, (idx, score) in enumerate(zip(idxs, scores), 1):
            if rank == n_pos + 1:
                print(" Contributing negatively towards the prediction:")
            res_char = seq[idx] if 0 <= idx < len(seq) else "?"
            bar_len  = int(40 * score / max_score) if max_score > 0 else 0
            bar      = "█" * abs(bar_len)
            print(f"  {rank:2d}. Pos {(idx+1):3d} ({res_char}): {bar} {score:.4f}")

        # Ablation validation
        if 'ablation_validation' in results:
            print("\n--- Ablation Validation ---")
            abl = results['ablation_validation'][sid]
            print(f"Original Loss: {abl['original_loss']:.4f}")
            print(f"Original Class: {abl['original_class']}")

            for k_name, metrics in abl['ablations'].items():
                print(f"\n  {k_name}:")
                print(f"    Loss increase: {metrics['loss_increase']:.4f}")
                print(f"    Predicted class after ablation: {metrics['predicted_class_after_abl']}")


from matplotlib.colors import LinearSegmentedColormap


def visualize_sequence_explanations(
    results, true_labels=None,
    class_names=['barrier', 'cation', 'anion'],
    max_sequences=10, figsize_per_seq=(22, 2),
    save_name=None
):
    """
    Visualise IG explanations per sequence as coloured heatmaps.
    Red = strong negative, white = neutral, green = strong positive.
    """
    samples     = results["samples"]
    num_to_plot = min(len(samples), max_sequences)

    cmap = LinearSegmentedColormap.from_list("neg_white_pos", ["red", "white", "green"])

    def normalize_attributions(attr_vec):
        max_abs = np.max(np.abs(attr_vec)) if attr_vec.size > 0 else 1.0
        return attr_vec / max_abs if max_abs != 0 else np.zeros_like(attr_vec)

    for i in range(num_to_plot):
        sample = samples[i]
        sid    = sample["sample_id"]
        seq    = sample["sequence"]
        L      = len(seq)
        desc   = sample["input_seq_desc"] or f"sample_{sid}"

        pred_class = sample["predicted_class"]
        true_class = true_labels[sid] if true_labels is not None else None
        pred_label = class_names[pred_class] if class_names else str(pred_class)
        true_label = class_names[true_class] if (class_names and true_class is not None) else None
        confidence = sample["confidence"]

        # Build full attribution vector and normalise to [-1, 1]
        full_attr = np.array(sample["attributions"].squeeze().cpu(), dtype=float)
        norm_attr = normalize_attributions(full_attr)
        colors    = cmap((norm_attr + 1) / 2.0)

        fig, ax = plt.subplots(figsize=figsize_per_seq, constrained_layout=True)
        ax.set_axis_off()

        # Header
        header = f"{desc} | true: {true_label} | " if true_labels is not None else f"{desc} | "
        header += f"pred: {pred_label}"
        if confidence is not None:
            header += f" | conf: {confidence:.3f}"
        ax.text(0.01, 1.1, header, fontsize=10, transform=ax.transAxes, ha="left", va="bottom")

        # Draw coloured residue blocks
        x_start  = 0.02
        x_step   = 0.96 / max(L, 1)
        y_center = 0.65
        y_index  = 0.50

        for pos in range(L):
            x = x_start + pos * x_step
            rect = plt.Rectangle(
                (x, y_center - 0.06), x_step * 0.9, 0.12,
                color=colors[pos], edgecolor=None,
                transform=ax.transAxes, clip_on=False,
            )
            ax.add_patch(rect)
            ax.text(x + x_step / 2, y_center, seq[pos], fontsize=9,
                    ha="center", va="center", transform=ax.transAxes, color="black")
            ax.text(x + x_step / 2, y_index, str(pos + 1), fontsize=7,
                    ha="center", va="center", rotation=90, transform=ax.transAxes,
                    color=(0.5, 0.5, 0.5, 0.6))

        # Colour bar legend
        cax = fig.add_axes([0.8, 0.05, 0.18, 0.06])
        grad = np.linspace(-1, 1, 256).reshape(1, -1)
        cax.imshow(grad, aspect="auto", cmap=cmap, extent=[-1, 1, 0, 1])
        cax.set_yticks([])
        cax.set_xticks([-1, 0, 1])
        cax.set_xticklabels(["neg", "0", "pos"], fontsize=7)
        cax.set_title("Contribution to Prediction", fontsize=7)

        if save_name:
            plt.savefig(f"{save_name}.png", bbox_inches='tight', dpi=300)
        
        plt.show()

def visualize_attention_explanations(
    attn_weights, input_seqs, input_seq_ids,
    predicted_classes, confidences,
    true_classes=None, class_names=['barrier', 'cation', 'anion'],
    max_sequences=10, figsize_per_seq=(22, 2),
    save_name=None
):
    """
    Visualise attention weights per sequence.
    White = low attention, purple = high attention.
    """
    num_to_plot = min(
        max_sequences, len(input_seqs),
        len(predicted_classes), len(confidences),
        attn_weights.shape[0],
    )

    cmap = LinearSegmentedColormap.from_list("low_high", ["white", "lavender", "purple"])

    for i in range(num_to_plot):
        attn = attn_weights[i].cpu().numpy()
        seq  = input_seqs[i]
        L    = len(seq)
        desc = input_seq_ids[i] or f"sample_{i}"

        pred_class = predicted_classes[i]
        true_class = true_classes[i] if true_classes is not None else None
        pred_label = class_names[pred_class] if class_names else str(pred_class)
        true_label = class_names[true_class] if (class_names and true_class is not None) else None
        confidence = confidences[i] if confidences is not None else None

        # Normalise attention to [0, 1]
        max_attn  = np.max(attn)
        norm_attn = attn / max_attn if max_attn != 0 else np.zeros_like(attn)
        colors    = cmap(norm_attn)

        fig, ax = plt.subplots(figsize=figsize_per_seq, constrained_layout=True)
        ax.set_axis_off()

        # Header
        header = f"{desc} | true: {true_label} | " if true_classes is not None else f"{desc} | "
        header += f"pred: {pred_label}"
        if confidence is not None:
            header += f" | conf: {confidence:.3f}"
        ax.text(0.01, 1.1, header, fontsize=10, transform=ax.transAxes, ha="left", va="bottom")

        # Draw coloured residue blocks
        x_start  = 0.02
        x_step   = 0.96 / max(L, 1)
        y_center = 0.65
        y_index  = 0.50

        for pos in range(L):
            x = x_start + pos * x_step
            rect = plt.Rectangle(
                (x, y_center - 0.06), x_step * 0.9, 0.12,
                color=colors[pos], edgecolor=None,
                transform=ax.transAxes, clip_on=False,
            )
            ax.add_patch(rect)
            ax.text(x + x_step / 2, y_center, seq[pos], fontsize=9,
                    ha="center", va="center", transform=ax.transAxes, color="black")
            ax.text(x + x_step / 2, y_index, str(pos + 1), fontsize=7,
                    ha="center", va="center", rotation=90, transform=ax.transAxes,
                    color=(0.5, 0.5, 0.5, 0.6))

        # Colour bar legend
        cax = fig.add_axes([0.8, 0.05, 0.18, 0.06])
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        cax.imshow(grad, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
        cax.set_yticks([])
        cax.set_xticks([0, 0.5, 1])
        cax.set_xticklabels(["low", "med", "high"], fontsize=7)
        cax.set_title("Attention Weight", fontsize=7)

        if save_name:
            plt.savefig(f"{save_name}.png", bbox_inches='tight', dpi=300)
        
        plt.show()
   

def compute_attention_weights(model, inputs):
    """Extract attention weights from the classifier (adversarial mode disabled)."""
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(inputs, return_attn=True)

    return logits, attn_weights

def compute_saliency(model, inputs):
    """Extract gradient saliency scores from the classifier (adversarial mode disabled)."""
    model.eval()

    inputs = inputs.requires_grad_(True)
    logits = model(inputs)
    predicted_classes = logits.argmax(dim=1)

    # Backprop the score of the predicted class for each sample
    scores = logits[torch.arange(len(predicted_classes)), predicted_classes]
    scores.sum().backward()

    # Collapse embedding dim → (B, L) by taking the absolute gradient magnitude
    saliency = inputs.grad.abs().mean(dim=-1)

    return logits, saliency.detach()