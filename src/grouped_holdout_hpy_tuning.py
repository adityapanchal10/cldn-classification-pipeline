# Grouped-holdout hyperparameter tuning (macro F1)
import itertools
import json
import os
import time
import hashlib
import numpy as np
import pandas as pd
import gc

from src.grouped_holdout import train_grouped_holdout_cv

def _config_hash(cfg):
    payload = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def _load_cache(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_cache(cache, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def _build_grid(space):
    keys = list(space.keys())
    for values in itertools.product(*[space[k] for k in keys]):
        cfg = dict(zip(keys, values))
        yield cfg

def _summarize_grouped_results(grouped_results):
    metrics_df = pd.DataFrame([{
        "accuracy": r["accuracy"],
        "macro_recall": r["macro_recall"],
        "macro_f1": r["macro_f1"],
        "recall_barrier": r["recall_barrier"],
        "recall_cation": r["recall_cation"],
        "recall_anion": r["recall_anion"],
    } for r in grouped_results])
    return {
        "macro_f1_mean": float(metrics_df["macro_f1"].mean()),
        "macro_f1_std": float(metrics_df["macro_f1"].std()),
        "macro_recall_mean": float(metrics_df["macro_recall"].mean()),
        "recall_cation_mean": float(metrics_df["recall_cation"].mean()),
        "recall_anion_mean": float(metrics_df["recall_anion"].mean()),
    }

def pick_best_config(df_results, default_hp):
    if df_results is None or df_results.empty:
        return dict(default_hp)
    best_cfg = df_results.iloc[0]["config"]
    merged = dict(default_hp)
    merged.update(best_cfg)
    return merged

def run_grouped_holdout_trial(cfg, cache, cache_path, file_meta=None, embeddings_by_file=None, model=None, seq_len=None, device="cuda"):
    key = _config_hash(cfg)
    if key in cache:
        return cache[key]

    grouped_results, _ = train_grouped_holdout_cv(
        file_meta=file_meta,
        embeddings_by_file=embeddings_by_file,
        model_fn=lambda: model(),
        seq_len=seq_len,
        num_epochs=cfg["num_epochs"],
        device=device,
        patience=cfg["patience"],
        batch_size=cfg["batch_size"],
        num_classes=3,
        class_names={0: "barrier", 1: "cation", 2: "anion"},
        useWeightedSampler=cfg["useWeightedSampler"],
        weighted_ce_power=cfg.get("weighted_ce_power", 1.0),
        warmup_epochs=cfg["warmup_epochs"],
        optimizer_lr=cfg["optimizer_lr"],
        optimizer_weight_decay=cfg["optimizer_weight_decay"],
        schedular_patience=cfg["schedular_patience"],
    )

    summary = _summarize_grouped_results(grouped_results)
    result = {"config": cfg, **summary}
    cache[key] = result
    _save_cache(cache, cache_path)
    return result

def run_grouped_holdout_hyperparameter_tuning(search_space, max_trails=12, cache_path="grouped_holdout_tuning_cache.json", save_path="grouped_holdout_tuning_results.csv", file_meta=None, embeddings_by_file=None, model=None, seq_len=None, device="cuda"):
    cache = _load_cache(cache_path)
    results = []

    for cfg in _build_grid(search_space):
        if cfg["useWeightedSampler"] and cfg["weighted_ce_power"] > 0.0:
            continue
        if cfg["weighted_ce_power"] < 0.0:
            cfg = {k: v for k, v in cfg.items() if k != "weighted_ce_power"}
        results.append(run_grouped_holdout_trial(cfg, cache, cache_path, file_meta, embeddings_by_file, model, seq_len, device))
        if len(results) >= max_trials:
            break
        time.sleep(0.1)
        # clear cache from memory to avoid memory bloat
        gc.collect()
        torch.cuda.empty_cache()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["macro_f1_mean"], ascending=False)
    df_results.to_csv("grouped_holdout_tuning_results.csv", index=False)

    BEST_CFG = pick_best_config(df_results, DEFAULT_HP)
    print("Best grouped-holdout config:")
    print(BEST_CFG)
    with open("grouped_holdout_best_config.json", "w", encoding="utf-8") as f:
        json.dump(BEST_CFG, f, indent=2)

    return BEST_CFG
    