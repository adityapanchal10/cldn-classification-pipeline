from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import esm
import torch
from Bio import SeqIO


def load_esm_model(model_name: str = "esm_msa1b_t12_100M_UR50S"):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    return model, alphabet


class ESMEmbedder:
    def __init__(self, model_name: str = "esm_msa1b_t12_100M_UR50S", device=None):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.model, self.alphabet = load_esm_model(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(self.device)
        self.valid_chars = set(self.alphabet.all_toks)
        self.model_name = model_name
        self.is_msa_model = "msa" in model_name.lower()
        self.repr_layer = None
        self.embed_dim = None
        self.pad_char = "-" if self.is_msa_model else "X"
        self._detect_model_properties()
        self.model.eval()

    def _clean_sequence(self, seq: str) -> str:
        return "".join(
            c if c in self.valid_chars else self.pad_char for c in seq
        ).upper()

    def pad_or_truncate(self, sequences, seq_length, pad_char=None):
        pad_char = self.pad_char if pad_char is None else pad_char
        return [
            (
                seq[:seq_length]
                if len(seq) > seq_length
                else seq.ljust(seq_length, pad_char)
            )
            for seq in sequences
        ]

    def _detect_model_properties(self):
        num_layers = None
        if hasattr(self.model, "num_layers"):
            try:
                num_layers = int(getattr(self.model, "num_layers"))
            except Exception:
                num_layers = None

        if num_layers is None and hasattr(self.model, "args"):
            for name in ("num_layers", "n_layer", "nlayers", "encoder_layers"):
                v = getattr(self.model.args, name, None)
                if v is not None:
                    try:
                        num_layers = int(v)
                        break
                    except Exception:
                        continue

        if num_layers is None and hasattr(self.model, "cfg"):
            for name in ("num_layers", "n_layer", "nlayers"):
                v = getattr(self.model.cfg, name, None)
                if v is not None:
                    try:
                        num_layers = int(v)
                        break
                    except Exception:
                        continue

        if num_layers is not None:
            self.repr_layer = num_layers
            print(f"Detected num_layers attribute: using repr_layer={self.repr_layer}")
        else:
            try:
                if self.is_msa_model:
                    _, _, tokens = self.batch_converter(
                        [[("_s0", "A" * 8), ("_s1", "A" * 8)]]
                    )
                else:
                    _, _, tokens = self.batch_converter([("_s0", "A" * 8)])
                tokens = tokens.to(self.device)
                with torch.no_grad():
                    results = self.model(tokens, repr_layers=[0], return_contacts=False)
                keys = list(results.get("representations", {}).keys())
                if keys:
                    int_keys = []
                    for k in keys:
                        try:
                            int_keys.append(int(k))
                        except Exception:
                            pass
                    if int_keys:
                        self.repr_layer = max(int_keys)
                        rep = results["representations"][self.repr_layer]
                        self.embed_dim = rep.shape[-1]
                        print(
                            f"Detected representation keys: {keys}; "
                            f"using repr_layer={self.repr_layer}, embed_dim={self.embed_dim}"
                        )
                if self.repr_layer is None:
                    self.repr_layer = getattr(self.model, "num_layers", 12)
                    print(f"Fallback repr_layer set to {self.repr_layer}")
                if self.embed_dim is None:
                    self.embed_dim = (
                        getattr(self.model, "embed_dim", None)
                        or (
                            getattr(self.model, "args", None)
                            and getattr(self.model.args, "embed_dim", None)
                        )
                        or 768
                    )
                    print(f"Fallback embed_dim set to {self.embed_dim}")
            except Exception:
                self.repr_layer = getattr(self.model, "num_layers", 12)
                self.embed_dim = (
                    getattr(self.model, "embed_dim", None)
                    or (
                        getattr(self.model, "args", None)
                        and getattr(self.model.args, "embed_dim", None)
                    )
                    or 768
                )

    def embed_msa(self, sequences, seq_length=190, max_msa_depth=300):
        if not self.is_msa_model:
            raise ValueError(
                "embed_msa() requires an MSA Transformer checkpoint such as esm_msa1b_t12_100M_UR50S."
            )

        sequences = [self._clean_sequence(s) for s in sequences]
        sequences = self.pad_or_truncate(sequences, seq_length)
        n_sequences = len(sequences)

        all_embeddings = []

        for start in range(0, n_sequences, max_msa_depth):
            chunk = sequences[start : start + max_msa_depth]
            msa_input = [(f"seq{start + j}", seq) for j, seq in enumerate(chunk)]

            _, _, batch_tokens = self.batch_converter([msa_input])
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(
                    batch_tokens, repr_layers=[self.repr_layer], return_contacts=False
                )

            token_emb = results["representations"][self.repr_layer]
            token_emb = token_emb[:, :, 1:, :]
            token_emb = token_emb.squeeze(0)
            all_embeddings.append(token_emb.cpu())

        output_embeddings = torch.cat(all_embeddings, dim=0)
        assert (
            len(output_embeddings.shape) == 3
        ), f"Unexpected shape: {output_embeddings.shape}"
        return output_embeddings

    def embed_sequences_per_residue(self, sequences, seq_length=190, batch_size=1):
        sequences = [self._clean_sequence(s) for s in sequences]
        sequences = self.pad_or_truncate(sequences, seq_length)

        all_embeddings = []
        total_batches = (len(sequences) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(sequences))
            batch = sequences[start:end]

            if self.is_msa_model:
                batch_inputs = [
                    [(f"seq{start + i}", seq)] for i, seq in enumerate(batch)
                ]
            else:
                batch_inputs = [(f"seq{start + i}", seq) for i, seq in enumerate(batch)]
            _, _, batch_tokens = self.batch_converter(batch_inputs)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(
                    batch_tokens, repr_layers=[self.repr_layer], return_contacts=False
                )

            token_emb = results["representations"][self.repr_layer]
            if self.is_msa_model:
                token_emb = token_emb[:, 0, 1:, :]
            else:
                token_emb = token_emb[:, 1:-1, :]
            all_embeddings.append(token_emb.cpu())

        output_embeddings = torch.cat(all_embeddings, dim=0)
        return output_embeddings


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records = []
    for record in SeqIO.parse(str(path), "fasta"):
        sequence = str(record.seq).replace(" ", "").replace("\n", "").upper()
        header = record.description.strip()
        records.append((header, sequence))
    return records


def discover_fasta_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        [
            path
            for path in input_path.rglob("*")
            if path.suffix.lower() in {".fa", ".fasta", ".faa", ".fna"}
        ]
    )


def split_records(
    records: List[Tuple[str, str]], test_fraction: float, seed: int
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    if not records:
        return [], []

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    test_size = (
        max(1, int(round(len(records) * test_fraction))) if len(records) > 1 else 0
    )
    test_indices = set(indices[:test_size])

    train_records = [
        record for idx, record in enumerate(records) if idx not in test_indices
    ]
    test_records = [record for idx, record in enumerate(records) if idx in test_indices]

    if not train_records and test_records:
        train_records = [test_records.pop()]

    return train_records, test_records


def extract_family(seq_id: str) -> str:
    parts = seq_id.split("|")
    for part in parts:
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip().lower()
            if key in {"major_label", "family", "label", "class", "gene"}:
                return value
    raise ValueError(f"Could not extract family from header: {seq_id}")


def group_records_by_family(
    records: List[Tuple[str, str]],
) -> Dict[str, List[Tuple[str, str]]]:
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for record in records:
        grouped[extract_family(record[0])].append(record)
    return dict(grouped)


def flatten_chunks(chunks: List[List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    return [record for chunk in chunks for record in chunk]


def assign_chunks_to_train_test(
    chunks: List[List[Tuple[str, str]]], test_fraction: float, seed: int
) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
    if not chunks:
        return [], []

    shuffled = list(chunks)
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, []

    n_test = int(round(len(shuffled) * test_fraction))
    n_test = max(1, min(len(shuffled) - 1, n_test))

    test_chunks = shuffled[:n_test]
    train_chunks = shuffled[n_test:]
    return train_chunks, test_chunks


def resolve_balanced_chunk_size(
    grouped: Dict[str, List[Tuple[str, str]]],
    requested_chunk_size: Optional[int],
    default_max_msa_depth: int,
) -> int:
    if not grouped:
        raise ValueError("No families found in the input records.")

    n_families = len(grouped)
    min_family_size = min(len(v) for v in grouped.values())

    if requested_chunk_size is None:
        per_family = min(default_max_msa_depth // n_families, min_family_size)
        if per_family < 1:
            raise ValueError(
                f"Cannot derive a valid balanced chunk size from "
                f"max_msa_depth={default_max_msa_depth} and {n_families} families."
            )
        return per_family * n_families

    if requested_chunk_size <= 0:
        raise ValueError("Balanced chunk size must be positive.")
    if requested_chunk_size % n_families != 0:
        raise ValueError(
            f"Balanced chunk size {requested_chunk_size} must be divisible by "
            f"the number of families ({n_families})."
        )

    per_family = requested_chunk_size // n_families
    if per_family > min_family_size:
        raise ValueError(
            f"Balanced chunk size {requested_chunk_size} requires {per_family} "
            f"sequences per family, but the smallest family has only {min_family_size}."
        )

    return requested_chunk_size


def make_balanced_chunks(
    records: List[Tuple[str, str]], chunk_size: int, seed: int
) -> Tuple[List[List[Tuple[str, str]]], List[Tuple[str, str]]]:
    grouped = group_records_by_family(records)
    rng = random.Random(seed)
    families = sorted(grouped)
    per_family = chunk_size // len(families)

    for fam in families:
        rng.shuffle(grouped[fam])

    chunks: List[List[Tuple[str, str]]] = []
    while all(len(grouped[fam]) >= per_family for fam in families):
        chunk_records: List[Tuple[str, str]] = []
        for fam in families:
            chunk_records.extend(grouped[fam][:per_family])
            grouped[fam] = grouped[fam][per_family:]
        rng.shuffle(chunk_records)
        chunks.append(chunk_records)

    leftovers = flatten_chunks([grouped[fam] for fam in families if grouped[fam]])
    return chunks, leftovers


def resolve_family_chunk_size(
    grouped: Dict[str, List[Tuple[str, str]]],
    requested_chunk_size: Optional[int],
    default_max_msa_depth: int,
) -> int:
    if not grouped:
        raise ValueError("No families found in the input records.")

    min_family_size = min(len(v) for v in grouped.values())

    if requested_chunk_size is None:
        chunk_size = min(default_max_msa_depth, min_family_size)
    else:
        chunk_size = requested_chunk_size

    if chunk_size <= 0:
        raise ValueError("Family chunk size must be positive.")
    if chunk_size > min_family_size:
        raise ValueError(
            f"Family chunk size {chunk_size} is larger than the smallest family size "
            f"({min_family_size})."
        )

    return chunk_size


def make_family_chunks(
    records: List[Tuple[str, str]], chunk_size: int, seed: int
) -> Tuple[List[List[Tuple[str, str]]], List[Tuple[str, str]]]:
    grouped = group_records_by_family(records)
    rng = random.Random(seed)

    chunks: List[List[Tuple[str, str]]] = []
    leftovers: List[Tuple[str, str]] = []

    for fam in sorted(grouped):
        fam_records = list(grouped[fam])
        rng.shuffle(fam_records)

        n_full_chunks = len(fam_records) // chunk_size
        for i in range(n_full_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunks.append(fam_records[start:end])

        leftovers.extend(fam_records[n_full_chunks * chunk_size :])

    rng.shuffle(chunks)
    return chunks, leftovers


def resolve_diverse_chunk_size(
    grouped: Dict[str, List[Tuple[str, str]]],
    requested_chunk_size: Optional[int],
) -> int:
    if not grouped:
        raise ValueError("No families found in the input records.")

    n_families = len(grouped)
    min_family_size = min(len(v) for v in grouped.values())
    total_sequences = sum(len(v) for v in grouped.values())

    if requested_chunk_size is None:
        # Auto mode: maximize diversity first, then quantity, while keeping
        # the maximum possible number of chunks.
        n_chunks = min_family_size
        return total_sequences // n_chunks

    if requested_chunk_size < n_families:
        raise ValueError(
            f"Diverse chunk size must be at least the number of families "
            f"({n_families}) so that no family is left out."
        )

    if requested_chunk_size > total_sequences:
        raise ValueError(
            f"Diverse chunk size {requested_chunk_size} exceeds the total number "
            f"of sequences ({total_sequences})."
        )

    feasible_n_chunks = min(min_family_size, total_sequences // requested_chunk_size)
    if feasible_n_chunks < 1:
        raise ValueError(
            f"Diverse chunk size {requested_chunk_size} is not feasible for this dataset."
        )

    return requested_chunk_size


def make_diverse_chunks(
    records: List[Tuple[str, str]], chunk_size: int, seed: int
) -> Tuple[List[List[Tuple[str, str]]], List[Tuple[str, str]]]:
    grouped = group_records_by_family(records)
    rng = random.Random(seed)
    families = sorted(grouped)

    if not families:
        return [], []

    n_families = len(families)
    min_family_size = min(len(v) for v in grouped.values())
    total_sequences = sum(len(v) for v in grouped.values())

    if chunk_size < n_families:
        raise ValueError(
            f"Diverse chunk size must be at least {n_families} "
            f"to include all families in every chunk."
        )

    n_chunks = min(min_family_size, total_sequences // chunk_size)
    if n_chunks < 1:
        raise ValueError(
            f"Diverse chunk size {chunk_size} is not feasible for this dataset."
        )

    for fam in families:
        rng.shuffle(grouped[fam])

    chunks: List[List[Tuple[str, str]]] = [[] for _ in range(n_chunks)]

    # Seed every chunk with one sequence from every family.
    remaining_by_family: Dict[str, List[Tuple[str, str]]] = {}
    for fam in families:
        fam_records = grouped[fam]
        seed_records = fam_records[:n_chunks]
        remaining_by_family[fam] = fam_records[n_chunks:]

        for idx, record in enumerate(seed_records):
            chunks[idx].append(record)

    # Fill remaining capacity in a round-robin way, prioritizing families
    # with the most leftover sequences.
    extra_pool: List[Tuple[str, str]] = []
    while True:
        available = [fam for fam in families if remaining_by_family[fam]]
        if not available:
            break

        available.sort(key=lambda fam: (-len(remaining_by_family[fam]), fam))
        for fam in available:
            if remaining_by_family[fam]:
                extra_pool.append(remaining_by_family[fam].pop())

    fill_order: List[int] = []
    extra_slots_per_chunk = chunk_size - n_families
    for _ in range(extra_slots_per_chunk):
        fill_order.extend(range(n_chunks))

    dropped_records: List[Tuple[str, str]] = []
    for idx, record in enumerate(extra_pool):
        if idx >= len(fill_order):
            dropped_records.append(record)
        else:
            chunks[fill_order[idx]].append(record)

    for chunk in chunks:
        rng.shuffle(chunk)

    return chunks, dropped_records


def embed_records_msa(
    embedder: ESMEmbedder,
    records: List[Tuple[str, str]],
    seq_length: int,
    chunk_size: int,
) -> torch.Tensor:
    if not records:
        return torch.empty(0)
    return embedder.embed_msa(
        [seq for _, seq in records],
        seq_length=seq_length,
        max_msa_depth=chunk_size,
    )


def embed_records_independent(
    embedder: ESMEmbedder,
    records: List[Tuple[str, str]],
    seq_length: int,
    batch_size: int,
) -> torch.Tensor:
    if not records:
        return torch.empty(0)
    return embedder.embed_sequences_per_residue(
        [seq for _, seq in records],
        seq_length=seq_length,
        batch_size=batch_size,
    )


def write_msa_artifact(
    output_dir: Path,
    fasta_path: Path,
    records: List[Tuple[str, str]],
    embedding: torch.Tensor,
    model_name: str,
    chunk_size: Optional[int] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{fasta_path.stem}.pt"
    payload = {
        "sequence_ids": [seq_id for seq_id, _ in records],
        "sequences": [seq for _, seq in records],
        "embedding": embedding,
        "meta": {
            "source_path": str(fasta_path),
            "embedding_mode": "msa",
            "split_name": "msa",
            "split_strategy": "per_file",
            "chunk_size": chunk_size,
            "model_name": model_name,
        },
    }
    torch.save(payload, artifact_path)
    return artifact_path


def write_independent_artifact(
    output_dir: Path,
    fasta_path: Path,
    records: List[Tuple[str, str]],
    embedding: torch.Tensor,
    model_name: str,
    chunk_size: Optional[int] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{fasta_path.stem}.pt"
    payload = {
        "sequence_ids": [seq_id for seq_id, _ in records],
        "sequences": [seq for _, seq in records],
        "embedding": embedding,
        "meta": {
            "source_path": str(fasta_path),
            "embedding_mode": "independent",
            "split_name": "independent",
            "split_strategy": "per_file",
            "chunk_size": chunk_size,
            "model_name": model_name,
        },
    }
    torch.save(payload, artifact_path)
    return artifact_path


def write_split_artifact(
    output_path: Path,
    source_name: str,
    split_name: str,
    records: List[Tuple[str, str]],
    embedding: torch.Tensor,
    model_name: str,
    embedding_mode: str,
    split_strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sequence_ids": [seq_id for seq_id, _ in records],
        "sequences": [seq for _, seq in records],
        "embedding": embedding,
        "meta": {
            "source_name": source_name,
            "split_name": split_name,
            "embedding_mode": embedding_mode,
            "split_strategy": split_strategy,
            "chunk_size": chunk_size,
            "model_name": model_name,
        },
    }
    torch.save(payload, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache ESM embeddings to Drive in a notebook-aligned format."
    )
    parser.add_argument(
        "--input", required=True, help="FASTA file or directory containing FASTA files."
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory where embeddings will be saved.",
    )
    parser.add_argument(
        "--embedding-mode", choices=["msa", "independent"], required=True
    )
    parser.add_argument("--model-name", default="esm_msa1b_t12_100M_UR50S")
    parser.add_argument("--seq-length", type=int, default=190)
    parser.add_argument("--max-msa-depth", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--msa-split-strategy",
        choices=["random", "balanced", "family", "diverse"],
        default="random",
        help="Single FASTA + msa mode only.",
    )
    parser.add_argument(
        "--msa-balanced-chunk-size",
        type=int,
        default=None,
        help=(
            "Balanced chunk size for single-file msa mode. "
            "If omitted, it is derived from max_msa_depth and family count."
        ),
    )
    parser.add_argument(
        "--msa-family-chunk-size",
        type=int,
        default=None,
        help=(
            "Family chunk size for single-file msa mode. "
            "If omitted, it is derived from max_msa_depth and the smallest family."
        ),
    )
    parser.add_argument(
        "--msa-diverse-chunk-size",
        type=int,
        default=None,
        help=(
            "Diverse chunk size for single-file msa mode. "
            "If omitted, it is set to the largest possible fixed size such that "
            "all chunks include every family at least once."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    embedder = ESMEmbedder(model_name=args.model_name, device=args.device)
    print(
        f"Using model {args.model_name} on device {embedder.device} "
        f"with seq_length={args.seq_length}"
    )

    fasta_files = discover_fasta_files(input_path)
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found under {input_path}")

    manifest_rows = []
    unseen_manifest_rows = []

    for fasta_path in fasta_files:
        records = read_fasta(fasta_path)
        if not records:
            continue

        sequences = [sequence for _, sequence in records]

        if input_path.is_file():
            if args.embedding_mode == "msa":
                strategy = args.msa_split_strategy
                best_chunk_records: List[Tuple[str, str]] = []
                dropped_records: List[Tuple[str, str]] = []

                if strategy == "random":
                    chunk_size = args.max_msa_depth
                    train_records, test_records = split_records(
                        records, test_fraction=args.test_fraction, seed=args.seed
                    )

                elif strategy == "balanced":
                    grouped = group_records_by_family(records)
                    chunk_size = resolve_balanced_chunk_size(
                        grouped=grouped,
                        requested_chunk_size=args.msa_balanced_chunk_size,
                        default_max_msa_depth=args.max_msa_depth,
                    )

                    balanced_chunks, dropped_records = make_balanced_chunks(
                        records=records,
                        chunk_size=chunk_size,
                        seed=args.seed,
                    )
                    if not balanced_chunks:
                        raise ValueError(
                            "Balanced strategy produced no full chunks. "
                            "Try a smaller balanced chunk size."
                        )

                    best_chunk_records = list(balanced_chunks[0])
                    train_chunks, test_chunks = assign_chunks_to_train_test(
                        balanced_chunks,
                        test_fraction=args.test_fraction,
                        seed=args.seed,
                    )
                    train_records = flatten_chunks(train_chunks)
                    test_records = flatten_chunks(test_chunks)

                elif strategy == "family":
                    grouped = group_records_by_family(records)
                    chunk_size = resolve_family_chunk_size(
                        grouped=grouped,
                        requested_chunk_size=args.msa_family_chunk_size,
                        default_max_msa_depth=args.max_msa_depth,
                    )

                    family_chunks, dropped_records = make_family_chunks(
                        records=records,
                        chunk_size=chunk_size,
                        seed=args.seed,
                    )
                    if not family_chunks:
                        raise ValueError(
                            "Family strategy produced no full chunks. "
                            "Try a smaller family chunk size."
                        )

                    train_chunks, test_chunks = assign_chunks_to_train_test(
                        family_chunks,
                        test_fraction=args.test_fraction,
                        seed=args.seed,
                    )
                    train_records = flatten_chunks(train_chunks)
                    test_records = flatten_chunks(test_chunks)

                elif strategy == "diverse":
                    grouped = group_records_by_family(records)
                    chunk_size = resolve_diverse_chunk_size(
                        grouped=grouped,
                        requested_chunk_size=args.msa_diverse_chunk_size,
                    )

                    diverse_chunks, dropped_records = make_diverse_chunks(
                        records=records,
                        chunk_size=chunk_size,
                        seed=args.seed,
                    )
                    if not diverse_chunks:
                        raise ValueError(
                            "Diverse strategy produced no full chunks. "
                            "Try a smaller diverse chunk size."
                        )

                    best_chunk_records = list(diverse_chunks[0])
                    train_chunks, test_chunks = assign_chunks_to_train_test(
                        diverse_chunks,
                        test_fraction=args.test_fraction,
                        seed=args.seed,
                    )
                    train_records = flatten_chunks(train_chunks)
                    test_records = flatten_chunks(test_chunks)

                else:
                    raise ValueError(f"Unsupported msa split strategy: {strategy}")

                train_embedding = embed_records_msa(
                    embedder=embedder,
                    records=train_records,
                    seq_length=args.seq_length,
                    chunk_size=chunk_size,
                )
                test_embedding = embed_records_msa(
                    embedder=embedder,
                    records=test_records,
                    seq_length=args.seq_length,
                    chunk_size=chunk_size,
                )

                train_path = write_split_artifact(
                    output_root / "train.pt",
                    fasta_path.stem,
                    "train",
                    train_records,
                    train_embedding,
                    args.model_name,
                    args.embedding_mode,
                    split_strategy=strategy,
                    chunk_size=chunk_size,
                )
                test_path = write_split_artifact(
                    output_root / "test.pt",
                    fasta_path.stem,
                    "test",
                    test_records,
                    test_embedding,
                    args.model_name,
                    args.embedding_mode,
                    split_strategy=strategy,
                    chunk_size=chunk_size,
                )

                manifest_rows.append(
                    [
                        str(fasta_path),
                        fasta_path.stem,
                        args.embedding_mode,
                        "train",
                        strategy,
                        chunk_size,
                        str(train_path),
                        ",".join(seq_id for seq_id, _ in train_records),
                    ]
                )
                manifest_rows.append(
                    [
                        str(fasta_path),
                        fasta_path.stem,
                        args.embedding_mode,
                        "test",
                        strategy,
                        chunk_size,
                        str(test_path),
                        ",".join(seq_id for seq_id, _ in test_records),
                    ]
                )

                if strategy in {"balanced", "diverse"} and best_chunk_records:
                    best_chunk_embedding = embed_records_msa(
                        embedder=embedder,
                        records=best_chunk_records,
                        seq_length=args.seq_length,
                        chunk_size=chunk_size,
                    )
                    best_chunk_path = write_split_artifact(
                        output_root / "best_chunk.pt",
                        fasta_path.stem,
                        "best_chunk",
                        best_chunk_records,
                        best_chunk_embedding,
                        args.model_name,
                        args.embedding_mode,
                        split_strategy=strategy,
                        chunk_size=chunk_size,
                    )
                    manifest_rows.append(
                        [
                            str(fasta_path),
                            fasta_path.stem,
                            args.embedding_mode,
                            "best_chunk",
                            strategy,
                            chunk_size,
                            str(best_chunk_path),
                            ",".join(seq_id for seq_id, _ in best_chunk_records),
                        ]
                    )

                if dropped_records:
                    unseen_embedding = embed_records_msa(
                        embedder=embedder,
                        records=dropped_records,
                        seq_length=args.seq_length,
                        chunk_size=len(dropped_records) if len(dropped_records) < 1024 else len(dropped_records) // 2,
                    )
                    unseen_path = write_split_artifact(
                        output_root / "unseen.pt",
                        fasta_path.stem,
                        "unseen",
                        dropped_records,
                        unseen_embedding,
                        args.model_name,
                        args.embedding_mode,
                        split_strategy=strategy,
                        chunk_size=len(dropped_records),
                    )
                    unseen_manifest_rows.append(
                        [
                            str(fasta_path),
                            fasta_path.stem,
                            args.embedding_mode,
                            "unseen",
                            strategy,
                            len(dropped_records),
                            str(unseen_path),
                            ",".join(seq_id for seq_id, _ in dropped_records),
                        ]
                    )

                print(
                    f"Processed {fasta_path.name} with msa strategy={strategy}, "
                    f"chunk_size={chunk_size}, "
                    f"train={len(train_records)}, test={len(test_records)}"
                )
                if strategy in {"balanced", "family", "diverse"} and dropped_records:
                    print(
                        f"Saved {len(dropped_records)} leftover records to unseen.pt "
                        f"to preserve uniform chunk size={chunk_size}."
                    )
                continue

            strategy = "random independent"
            chunk_size = args.batch_size

            train_records, test_records = split_records(
                records, test_fraction=args.test_fraction, seed=args.seed
            )

            train_embedding = embed_records_independent(
                embedder=embedder,
                records=train_records,
                seq_length=args.seq_length,
                batch_size=args.batch_size,
            )
            test_embedding = embed_records_independent(
                embedder=embedder,
                records=test_records,
                seq_length=args.seq_length,
                batch_size=args.batch_size,
            )

            train_path = write_split_artifact(
                output_root / "train.pt",
                fasta_path.stem,
                "train",
                train_records,
                train_embedding,
                args.model_name,
                args.embedding_mode,
                split_strategy=strategy,
                chunk_size=chunk_size,
            )
            test_path = write_split_artifact(
                output_root / "test.pt",
                fasta_path.stem,
                "test",
                test_records,
                test_embedding,
                args.model_name,
                args.embedding_mode,
                split_strategy=strategy,
                chunk_size=chunk_size,
            )

            manifest_rows.append(
                [
                    str(fasta_path),
                    fasta_path.stem,
                    args.embedding_mode,
                    "train",
                    strategy,
                    chunk_size,
                    str(train_path),
                    ",".join(seq_id for seq_id, _ in train_records),
                ]
            )
            manifest_rows.append(
                [
                    str(fasta_path),
                    fasta_path.stem,
                    args.embedding_mode,
                    "test",
                    strategy,
                    chunk_size,
                    str(test_path),
                    ",".join(seq_id for seq_id, _ in test_records),
                ]
            )

            print(f"Processed {fasta_path.name} into train/test")
            continue

        if args.embedding_mode == "msa":
            chunk_size = args.max_msa_depth
            embeddings = embedder.embed_msa(
                sequences, seq_length=args.seq_length, max_msa_depth=chunk_size
            )
            artifact_path = write_msa_artifact(
                output_root / "msa",
                fasta_path,
                records,
                embeddings,
                args.model_name,
                chunk_size=chunk_size,
            )
            manifest_rows.append(
                [
                    str(fasta_path),
                    fasta_path.stem,
                    "msa",
                    "msa",
                    "per_file",
                    chunk_size,
                    str(artifact_path),
                    ",".join(seq_id for seq_id, _ in records),
                ]
            )
        else:
            chunk_size = args.batch_size
            embeddings = embedder.embed_sequences_per_residue(
                sequences, seq_length=args.seq_length, batch_size=chunk_size
            )
            artifact_path = write_independent_artifact(
                output_root / "independent",
                fasta_path,
                records,
                embeddings,
                args.model_name,
                chunk_size=chunk_size,
            )
            manifest_rows.append(
                [
                    str(fasta_path),
                    fasta_path.stem,
                    "independent",
                    "independent",
                    "per_file",
                    chunk_size,
                    str(artifact_path),
                    ",".join(seq_id for seq_id, _ in records),
                ]
            )

        print(f"Processed {fasta_path.name}")

    manifest_path = output_root / f"manifest_{args.embedding_mode}.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_path",
                "item_id",
                "embedding_mode",
                "split_name",
                "split_strategy",
                "chunk_size",
                "artifact_path",
                "sequence_id",
            ]
        )
        writer.writerows(manifest_rows)

    print(f"Wrote manifest to {manifest_path}")

    if unseen_manifest_rows:
        unseen_manifest_path = output_root / "manifest_unseen.csv"
        with open(unseen_manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "source_path",
                    "item_id",
                    "embedding_mode",
                    "split_name",
                    "split_strategy",
                    "chunk_size",
                    "artifact_path",
                    "sequence_id",
                ]
            )
            writer.writerows(unseen_manifest_rows)

        print(f"Wrote unseen manifest to {unseen_manifest_path}")


if __name__ == "__main__":
    main()