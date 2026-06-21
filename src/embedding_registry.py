from pathlib import Path


def get_embedding_dir(cfg, unseen=False):
    dataset_type = cfg["data"]["dataset_type"]
    split_mode = cfg["evaluation"]["mode"]
    msa_mode = cfg["embedder"].get("use_msa_mode", False)
    embedding_mode = "msa" if msa_mode else "independent"
    if embedding_mode == "msa" and split_mode == "single":
        distribution = cfg["data"]["split_strategy"]
        embedding_split = f"msa/{distribution}"
    else:
        embedding_split = embedding_mode

    if unseen:
        base_dir = Path("/content/drive/MyDrive/Thesis data/embeddings/msa_transformer")
        return (
            base_dir
            / dataset_type
            / split_mode
            / embedding_split
            / f"manifest_unseen.csv"
        )

    embedder_name = cfg["embedder"].get("name")
    if embedder_name == "esm2":
        base_dir = Path("/content/drive/MyDrive/Thesis data/embeddings/esm2")
        embedding_dir = base_dir / dataset_type / f"manifest_independent.csv"
    else:
        base_dir = Path("/content/drive/MyDrive/Thesis data/embeddings/msa_transformer")
        embedding_dir = (
            base_dir
            / dataset_type
            / split_mode
            / embedding_split
            / f"manifest_{embedding_mode}.csv"
        )

    return embedding_dir

def get_reference_fasta_path(cfg):
    dataset_type = cfg["data"]["dataset_type"]
    eval_mode = cfg["evaluation"]["mode"]
    split_strategy = cfg["data"]["split_strategy"]
    is_msa = cfg["embedder"].get("use_msa_mode", False)

    if eval_mode=="single" and split_strategy in ["diverse", "balanced"] and is_msa:
        base_dir = Path("/content/drive/MyDrive/Thesis data/embeddings/msa_transformer")
        reference_fasta_path = base_dir / dataset_type / eval_mode / "msa" / split_strategy / "best_chunk.fasta"
        return reference_fasta_path
    else:
        raise ValueError("Reference fasta path is only defined for single evaluation mode with diverse or balanced split strategy and msa mode enabled.")
