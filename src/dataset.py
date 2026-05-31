import torch
import numpy as np
from Bio import SeqIO
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MSADataset:
    """
    Loads multiple MSA files and exposes sequences with file-level metadata.
    No train/val split here — splitting is handled by LOFO-CV or the all-data loader.
    """

    def __init__(self, msa_files, labels, test_data=False):
        assert len(msa_files) == len(labels), "msa_files and labels must be same length"
        self.msa_files = msa_files
        self.labels = labels
        self.test_data = test_data

        self.combined_ids = []
        self.combined_seqs = []
        self.combined_labels = []
        self.combined_file_indices = []  # Integer index into msa_files
        self.combined_file_names = []  # Human-readable file name

        self._process_files()

    def _process_files(self):
        for file_idx, (msa_file, label) in enumerate(zip(self.msa_files, self.labels)):
            records = list(SeqIO.parse(msa_file, format="fasta"))
            fname = msa_file.split("/")[-1].replace(".fasta", "")
            for rec in records:
                self.combined_ids.append(rec.description)
                self.combined_seqs.append(str(rec.seq))
                self.combined_labels.append(label)
                self.combined_file_indices.append(file_idx)
                self.combined_file_names.append(fname)

    def getSequences(self):
        if self.test_data:
            return self.combined_ids, self.combined_seqs
        return self.combined_seqs, self.combined_labels

    def getSequenceLength(self):
        lengths = set(len(s) for s in self.combined_seqs)
        assert len(lengths) == 1, f"Inconsistent sequence lengths: {lengths}"
        return lengths.pop()

    def getTotalSequences(self):
        return len(self.combined_seqs)

    def getFileGroups(self):
        """Returns (file_index_array, file_name_array) aligned with sequences."""
        return (
            np.array(self.combined_file_indices),
            np.array(self.combined_file_names),
        )

    def getMSAsByFile(self):
        """
        Returns a dict: file_idx -> list of sequences.
        Used for true MSA-mode embedding (all sequences from one file together).
        """
        msa_dict = {}
        for id, seq, label, file_idx, fname in zip(
            self.combined_ids,
            self.combined_seqs,
            self.combined_labels,
            self.combined_file_indices,
            self.combined_file_names,
        ):
            if file_idx not in msa_dict:
                msa_dict[file_idx] = {
                    "ids": [],
                    "seqs": [],
                    "label": label,
                    "fname": fname,
                }
            msa_dict[file_idx]["seqs"].append(seq)
            msa_dict[file_idx]["ids"].append(id)
        return msa_dict


# Label mapping: 0 = Barrier forming, 1 = Cation-channel, 2 = Anion-channel
CLASS_MAP = {
    0: "Barrier forming",
    1: "Cation-channel forming",
    2: "Anion-channel forming",
}
