from pathlib import Path


def get_embedding_dir(cfg):

    dataset_type = cfg["data"]["dataset_type"]

    split_mode = cfg["evaluation"]["mode"]

    msa_mode = cfg["data"]["use_msa_mode"]

    embedding_mode = "msa" if msa_mode else "independent"

    base_dir = Path("/content/drive/MyDrive/Thesis data/embeddings")

    embedding_dir = (
        base_dir
        / dataset_type
        / split_mode
        / embedding_mode
    )

    return embedding_dir