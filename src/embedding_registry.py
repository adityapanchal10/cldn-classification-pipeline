from pathlib import Path

def get_embedding_dir(cfg):
    dataset_type = cfg["data"]["dataset_type"]
    split_mode = cfg["evaluation"]["mode"]
    msa_mode = cfg["embedder"].get("use_msa_mode", False)
    embedding_mode = "msa" if msa_mode else "independent"

    embedder_name = cfg["embedder"].get("name")
    if embedder_name == "esm2":
        base_dir = Path("/content/drive/MyDrive/Thesis data/embeddings/esm2")
        embedding_dir = base_dir / dataset_type / f"manifest_{embedding_mode}.csv"
    else:
        base_dir = Path("/content/drive/MyDrive/Thesis data/embeddings/msa_transformer")
        embedding_dir = (
            base_dir
            / dataset_type
            / split_mode
            / embedding_mode
            / f"manifest_{embedding_mode}.csv"
        )

    return embedding_dir