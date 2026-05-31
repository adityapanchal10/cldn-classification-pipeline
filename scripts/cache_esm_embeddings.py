from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from Bio import SeqIO

import esm


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
        # Try several common attribute locations for layer count / embed dim
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

        # If we found a candidate, use it; otherwise run a tiny forward to inspect repr keys
        if num_layers is not None:
            self.repr_layer = num_layers
            print(f"Detected num_layers attribute: using repr_layer={self.repr_layer}")
        else:
            # minimal safe forward to inspect available representation layers and embedding dim
            try:
                if self.is_msa_model:
                    labels, strs, tokens = self.batch_converter(
                        [[("_s0", "A" * 8), ("_s1", "A" * 8)]]
                    )
                else:
                    labels, strs, tokens = self.batch_converter([("_s0", "A" * 8)])
                tokens = tokens.to(self.device)
                with torch.no_grad():
                    results = self.model(tokens, repr_layers=[0], return_contacts=False)
                keys = list(results.get("representations", {}).keys())
                if keys:
                    # keys are integers (layer ids) or strings; coerce to ints where possible
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
                            f"Detected representation keys: {keys}; using repr_layer={self.repr_layer}, embed_dim={self.embed_dim}"
                        )
                # fallback defaults
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
                # final fallback
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
        records.append((record.id, sequence))
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


def write_msa_artifact(
    output_dir: Path,
    fasta_path: Path,
    records: List[Tuple[str, str]],
    embedding: torch.Tensor,
    model_name: str,
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
    parser.add_argument("--max-msa-depth", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    embedder = ESMEmbedder(model_name=args.model_name, device=args.device)
    print(f"Using model {args.model_name} on device {embedder.device}")
    fasta_files = discover_fasta_files(input_path)
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found under {input_path}")

    manifest_rows = []
    for fasta_path in fasta_files:
        records = read_fasta(fasta_path)
        if not records:
            continue

        sequences = [sequence for _, sequence in records]

        if input_path.is_file():
            train_records, test_records = split_records(
                records, test_fraction=args.test_fraction, seed=args.seed
            )

            if args.embedding_mode == "msa":
                train_embedding = embedder.embed_msa(
                    [seq for _, seq in train_records],
                    seq_length=args.seq_length,
                    max_msa_depth=args.max_msa_depth,
                )
                test_embedding = (
                    embedder.embed_msa(
                        [seq for _, seq in test_records],
                        seq_length=args.seq_length,
                        max_msa_depth=args.max_msa_depth,
                    )
                    if test_records
                    else torch.empty(0)
                )
            else:
                train_embedding = embedder.embed_sequences_per_residue(
                    [seq for _, seq in train_records],
                    seq_length=args.seq_length,
                    batch_size=args.batch_size,
                )
                test_embedding = (
                    embedder.embed_sequences_per_residue(
                        [seq for _, seq in test_records],
                        seq_length=args.seq_length,
                        batch_size=args.batch_size,
                    )
                    if test_records
                    else torch.empty(0)
                )

            train_path = write_split_artifact(
                output_root / "train.pt",
                fasta_path.stem,
                "train",
                train_records,
                train_embedding,
                args.model_name,
                args.embedding_mode,
            )
            test_path = (
                write_split_artifact(
                    output_root / "test.pt",
                    fasta_path.stem,
                    "test",
                    test_records,
                    test_embedding,
                    args.model_name,
                    args.embedding_mode,
                )
                if test_records
                else None
            )

            manifest_rows.append(
                [
                    str(fasta_path),
                    fasta_path.stem,
                    "train",
                    str(train_path),
                    ",".join(seq_id for seq_id, _ in train_records),
                ]
            )
            if test_path is not None:
                manifest_rows.append(
                    [
                        str(fasta_path),
                        fasta_path.stem,
                        "test",
                        str(test_path),
                        ",".join(seq_id for seq_id, _ in test_records),
                    ]
                )

            print(f"Processed {fasta_path.name} into train/test")
            continue

        if args.embedding_mode == "msa":
            embeddings = embedder.embed_msa(
                sequences, seq_length=args.seq_length, max_msa_depth=args.max_msa_depth
            )
            artifact_path = write_msa_artifact(
                output_root / "msa", fasta_path, records, embeddings, args.model_name
            )
            manifest_rows.append(
                [
                    str(fasta_path),
                    fasta_path.stem,
                    "msa",
                    str(artifact_path),
                    "|".join(seq_id for seq_id, _ in records),
                ]
            )
        else:
            embeddings = embedder.embed_sequences_per_residue(
                sequences, seq_length=args.seq_length, batch_size=args.batch_size
            )
            artifact_path = write_independent_artifact(
                output_root / "independent",
                fasta_path,
                records,
                embeddings,
                args.model_name,
            )
            manifest_rows.append(
                [
                    str(fasta_path),
                    fasta_path.stem,
                    "independent",
                    str(artifact_path),
                    "|".join(seq_id for seq_id, _ in records),
                ]
            )

        print(f"Processed {fasta_path.name}")

    manifest_path = output_root / f"manifest_{args.embedding_mode}.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source_path", "item_id", "embedding_mode", "artifact_path", "sequence_id"]
        )
        writer.writerows(manifest_rows)

    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
