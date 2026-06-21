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
