from pathlib import Path
import csv
import sys
from collections import Counter
import torch
from Bio import SeqIO
import numpy as np

from src.dataset import MSADataset
from src.embedder import MSAEmbedder

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # Some platforms don't accept sys.maxsize directly
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10


def infer_label_from_item_id(item_id: str) -> int:
    key = item_id.strip().lower()

    if key.endswith("_barrier"):
        return 0
    if key.endswith("_cation"):
        return 1
    if key.endswith("_anion"):
        return 2

    cldn_label_map = {
        "cldn1": 0,
        "cldn2": 1,
        "cldn3": 0,
        "cldn5": 0,
        "cldn10a": 2,
        "cldn10b": 1,
        "cldn14": 0,
        "cldn15": 1,
        "cldn17": 2,
    }
    if key in cldn_label_map:
        return cldn_label_map[key]

    raise ValueError(f"Cannot infer label for item_id='{item_id}'")


def normalize_source_path(source_path: str) -> str:
    raw = (source_path or "").strip()
    if not raw:
        raise ValueError("Manifest row has empty source_path")

    p = Path(raw)
    if p.exists():
        return str(p)

    if raw.startswith("/content/drive/"):
        alt = "drive/" + raw[len("/content/drive/") :]
        if Path(alt).exists():
            return alt

    if raw.startswith("drive/"):
        alt = "/content/" + raw
        if Path(alt).exists():
            return alt

    return raw


def find_manifest_path(manifest_candidates):
    manifest_path = next(
        (Path(p) for p in manifest_candidates if Path(p).exists()), None
    )
    if manifest_path is None:
        raise FileNotFoundError(
            "Could not find manifest file in provided candidates: "
            + str(manifest_candidates)
        )
    return manifest_path


def resolve_artifact_path(row, manifest_file):
    raw_artifact = (row.get("artifact_path") or "").strip()
    artifact_name = Path(raw_artifact).name if raw_artifact else f"{row['item_id']}.pt"
    local_default = manifest_file.parent / artifact_name

    candidates = [
        Path(raw_artifact) if raw_artifact else local_default,
        manifest_file.parent / artifact_name,
        local_default,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return local_default


def to_tensor_artifact(obj, artifact_path):
    if torch.is_tensor(obj):
        return obj.float()

    if isinstance(obj, dict):
        if "embeddings" in obj and torch.is_tensor(obj["embeddings"]):
            return obj["embeddings"].float()
        if "tensor" in obj and torch.is_tensor(obj["tensor"]):
            return obj["tensor"].float()
        first_tensor = next((v for v in obj.values() if torch.is_tensor(v)), None)
        if first_tensor is not None:
            return first_tensor.float()

    return torch.tensor(obj, dtype=torch.float32)


def read_manifest_rows(manifest_path, unique_item_ids=False):
    rows = []
    seen_item_ids = set()

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                continue
            row["item_id"] = item_id

            if unique_item_ids and item_id in seen_item_ids:
                continue
            if unique_item_ids:
                seen_item_ids.add(item_id)

            rows.append(row)

    if not rows:
        raise ValueError(f"No usable rows found in manifest: {manifest_path}")

    return rows


def infer_label_from_header(header_text: str) -> int:
    lower_header = header_text.lower()
    marker = "major_label="
    if marker in lower_header:
        label_token = lower_header.split(marker, 1)[1].split()[0].split("|")[0].strip()
        return infer_label_from_item_id(label_token)

    raise ValueError(f"Could not infer label from header: {header_text}")


def load_grouped_embeddings_from_manifest(
    manifest_candidates,
    init_embedder=True,
    device=None,
):
    """
    Load grouped embeddings from a grouped-manifest CSV.
    Returns all grouped tensors/metadata used by downstream training code.
    """
    manifest_path = find_manifest_path(manifest_candidates)
    print(f"Using grouped manifest: {manifest_path}")

    manifest_entries = read_manifest_rows(manifest_path, unique_item_ids=True)
    manifest_by_item = {row["item_id"]: row for row in manifest_entries}

    msa_files = []
    file_labels = []
    for row in manifest_entries:
        msa_files.append(normalize_source_path(row.get("source_path", "")))
        file_labels.append(infer_label_from_item_id(row["item_id"]))

    print(f"Loaded {len(msa_files)} grouped MSA files from manifest")
    print(f"Label distribution (file-level): {Counter(file_labels)}")

    data = MSADataset(msa_files, file_labels)
    seq_len = data.getSequenceLength()
    print(f"Sequence length: {seq_len}")

    embedder = MSAEmbedder() if init_embedder else None

    msa_by_file = data.getMSAsByFile()
    file_meta = {}
    for file_idx in sorted(msa_by_file.keys()):
        info = msa_by_file[file_idx]
        file_meta[file_idx] = {
            "fname": info["fname"],
            "label": info["label"],
            "seqs": info["seqs"],
            "ids": info["ids"],
        }

    all_embeddings_list = []
    all_labels_list = []
    all_file_idx_list = []
    all_file_name_list = []
    embeddings_by_file = {}

    for file_idx in sorted(msa_by_file.keys()):
        info = msa_by_file[file_idx]
        seqs = info["seqs"]
        label = info["label"]
        fname = info["fname"]

        if fname not in manifest_by_item:
            raise KeyError(f"Missing manifest entry for item_id='{fname}'")

        row = manifest_by_item[fname]
        artifact_path = resolve_artifact_path(row, manifest_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {artifact_path}")

        emb_obj = torch.load(artifact_path, map_location="cpu")
        emb = to_tensor_artifact(emb_obj, artifact_path)

        if emb.ndim != 3:
            raise ValueError(
                f"Expected (N, L, D) embedding tensor, got {tuple(emb.shape)} for {artifact_path}"
            )
        if emb.shape[0] != len(seqs):
            raise ValueError(
                f"Row count mismatch for {fname}: embeddings={emb.shape[0]} vs sequences={len(seqs)}"
            )

        print(f"Loaded {fname}: {tuple(emb.shape)} from {artifact_path}")

        embeddings_by_file[file_idx] = emb
        all_embeddings_list.append(emb)
        all_labels_list.extend([label] * len(seqs))
        all_file_idx_list.extend([file_idx] * len(seqs))
        all_file_name_list.extend([fname] * len(seqs))

    all_embeddings = torch.cat(all_embeddings_list, dim=0)
    all_labels = torch.tensor(all_labels_list, dtype=torch.long)
    all_file_idx = np.array(all_file_idx_list)
    all_file_names = np.array(all_file_name_list)

    print(f"\nTotal embeddings: {all_embeddings.shape}")
    print(f"Class distribution: {Counter(all_labels.numpy())}")

    return {
        "manifest_path": manifest_path,
        "manifest_entries": manifest_entries,
        "manifest_by_item": manifest_by_item,
        "msa_files": msa_files,
        "file_labels": file_labels,
        "data": data,
        "seq_len": seq_len,
        "embedder": embedder,
        "file_meta": file_meta,
        "embeddings_by_file": embeddings_by_file,
        "all_embeddings": all_embeddings,
        "all_labels": all_labels,
        "all_file_idx": all_file_idx,
        "all_file_names": all_file_names,
    }


def load_single_embeddings_from_manifest(manifest_candidates):
    """
    Load single-manifest data and return ONLY train/val fields:
      - train_embeddings, val_embeddings
      - train_labels, val_labels
      - train_headers, val_headers
      - train_sequences, val_sequences
    Labels/headers/sequences are resolved by matching manifest sequence IDs to
    records in the source FASTA referenced by each manifest row.
    """
    manifest_path = find_manifest_path(manifest_candidates)
    print(f"Using single manifest: {manifest_path}")

    manifest_entries = read_manifest_rows(manifest_path, unique_item_ids=False)

    # Cache parsed FASTA records by normalized source path.
    fasta_record_cache = {}

    split_payload = {
        "train": {
            "embeddings": [],
            "labels": [],
            "headers": [],
            "sequences": [],
        },
        "val": {
            "embeddings": [],
            "labels": [],
            "headers": [],
            "sequences": [],
        },
    }

    for row in manifest_entries:
        raw_split = (row.get("split_name") or "unknown").strip().lower()
        split_key = (
            "val"
            if raw_split in {"val", "test", "unseen"}
            else "train" if raw_split == "train" else None
        )
        if split_key is None:
            continue

        chunk_size_str = (row.get("chunk_size") or "").strip()
        chunk_size = int(chunk_size_str) if chunk_size_str.isdigit() else None

        artifact_path = resolve_artifact_path(row, manifest_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {artifact_path}")

        emb_obj = torch.load(artifact_path, map_location="cpu")
        emb = to_tensor_artifact(emb_obj, artifact_path)
        if emb.ndim != 3:
            raise ValueError(
                f"Expected (N, L, D) tensor for single split, got {tuple(emb.shape)} for {artifact_path}"
            )

        source_path = normalize_source_path(row.get("source_path", ""))
        if source_path not in fasta_record_cache:
            id_to_record = {}

            for rec in SeqIO.parse(source_path, "fasta"):
                # Extract "q1" from:
                # "q1 asdfdgsfdsg | major_label=cldn1"
                seq_id = rec.description.split()[0].strip()

                if seq_id in id_to_record:
                    raise ValueError(
                        f"Duplicate sequence ID '{seq_id}' found in {source_path}"
                    )

                id_to_record[seq_id] = rec

            fasta_record_cache[source_path] = id_to_record

        id_to_record = fasta_record_cache[source_path]

        seq_field = (row.get("sequence_id") or "").strip()
        seq_ids = [s.strip().split("|")[0] for s in seq_field.split(",") if s.strip()]
        if not seq_ids:
            raise ValueError(
                f"No sequence IDs found for row in manifest: {manifest_path}"
            )

        if emb.shape[0] != len(seq_ids):
            raise ValueError(
                f"Single split count mismatch ({split_key}): embeddings={emb.shape[0]} vs sequence_ids={len(seq_ids)}"
            )

        row_labels = []
        row_headers = []
        row_sequences = []
        missing_ids = []

        for sid in seq_ids:
            rec = id_to_record.get(sid)
            if rec is None:
                missing_ids.append(sid)
                continue

            header = rec.description
            seq = str(rec.seq)
            label = infer_label_from_header(header)

            row_headers.append(header)
            row_sequences.append(seq)
            row_labels.append(label)

        if missing_ids:
            preview = missing_ids[:5]
            raise KeyError(
                f"Missing {len(missing_ids)} sequence IDs in FASTA for split '{split_key}'. Example: {preview}"
            )

        split_payload[split_key]["embeddings"].append(emb)
        split_payload[split_key]["labels"].extend(row_labels)
        split_payload[split_key]["headers"].extend(row_headers)
        split_payload[split_key]["sequences"].extend(row_sequences)

        print(
            f"Loaded single split '{split_key}': {tuple(emb.shape)} from {artifact_path}"
        )

    def _stack_or_empty(tensors):
        if not tensors:
            return torch.empty(0, 0, 0, dtype=torch.float32)
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)

    train_embeddings = _stack_or_empty(split_payload["train"]["embeddings"])
    val_embeddings = _stack_or_empty(split_payload["val"]["embeddings"])
    train_labels = torch.tensor(split_payload["train"]["labels"], dtype=torch.long)
    val_labels = torch.tensor(split_payload["val"]["labels"], dtype=torch.long)
    train_headers = split_payload["train"]["headers"]
    val_headers = split_payload["val"]["headers"]
    train_sequences = split_payload["train"]["sequences"]
    val_sequences = split_payload["val"]["sequences"]

    if train_embeddings.shape[0] != train_labels.shape[0]:
        raise ValueError(
            f"Train size mismatch: embeddings={train_embeddings.shape[0]} vs labels={train_labels.shape[0]}"
        )
    if val_embeddings.shape[0] != val_labels.shape[0]:
        raise ValueError(
            f"Val size mismatch: embeddings={val_embeddings.shape[0]} vs labels={val_labels.shape[0]}"
        )
    if train_embeddings.shape[0] != len(train_headers) or train_embeddings.shape[
        0
    ] != len(train_sequences):
        raise ValueError("Train embeddings/header/sequence counts are inconsistent")
    if val_embeddings.shape[0] != len(val_headers) or val_embeddings.shape[0] != len(
        val_sequences
    ):
        raise ValueError("Val embeddings/header/sequence counts are inconsistent")

    print(f"Single train embeddings: {tuple(train_embeddings.shape)}")
    print(f"Single val embeddings: {tuple(val_embeddings.shape)}")
    print(
        f"Single train labels: {Counter(train_labels.numpy()) if train_labels.numel() > 0 else {}}"
    )
    print(
        f"Single val labels: {Counter(val_labels.numpy()) if val_labels.numel() > 0 else {}}"
    )
    print(f"Chunk size (if any): {chunk_size}")

    return {
        "manifest_path": manifest_path,
        "manifest_entries": manifest_entries,
        "train_embeddings": train_embeddings,
        "val_embeddings": val_embeddings,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "train_headers": train_headers,
        "val_headers": val_headers,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "chunk_size": chunk_size,
    }
