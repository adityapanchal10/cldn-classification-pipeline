import torch
import numpy as np
from Bio import SeqIO
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os
import shutil
import subprocess
import tempfile

import numpy as np
from Bio import SeqIO


class MSADataset:
    """
    Loads multiple MSA files and exposes sequences with file-level metadata.

    If reference_msa is supplied, every MSA is first aligned to the reference
    alignment using MAFFT --add so that all sequences share the same coordinate
    system as the reference.
    """

    def __init__(
        self,
        msa_files,
        labels,
        test_data=False,
        reference_msa=None,
    ):
        assert len(msa_files) == len(labels), \
            "msa_files and labels must be same length"

        self.labels = labels
        self.test_data = test_data

        if reference_msa is not None:
            self.msa_files = self._align_to_reference(
                msa_files,
                reference_msa,
            )
        else:
            self.msa_files = msa_files

        self.combined_ids = []
        self.combined_seqs = []
        self.combined_labels = []
        self.combined_file_indices = []
        self.combined_file_names = []
        self.unseen_masks = {}

        self._process_files()

    def _align_to_reference(self, msa_files, reference_msa):
        """
        Align every supplied MSA to a fixed reference alignment using MAFFT.

        The output alignment contains BOTH:
            - Reference sequences
            - Newly-added sequences

        A boolean mask is stored for each aligned MSA indicating which rows
        correspond to the newly-added sequences.

        Returns
        -------
        list[str]
            Paths to the aligned MSA files.
        """

        if shutil.which("mafft") is None:
            raise RuntimeError(
                "MAFFT was not found. Install it first with:\n"
                "!apt-get install -y mafft"
            )

        reference_records = list(SeqIO.parse(reference_msa, "fasta"))
        reference_count = len(reference_records)

        aligned_files = []

        for file_idx, msa_file in enumerate(msa_files):

            # Count sequences in the unseen MSA
            unseen_records = list(SeqIO.parse(msa_file, "fasta"))
            unseen_count = len(unseen_records)

            # Save the output fasta where reference MSA is
            ref_msa_dir = os.path.dirname(reference_msa)
            outfile = os.path.join(
                ref_msa_dir,
                f"aligned_{os.path.basename(msa_file)}"
            )


            cmd = [
                "mafft",
                "--quiet",
                "--add",
                msa_file,
                reference_msa,
            ]

            with open(outfile, "w") as f:
                subprocess.run(
                    cmd,
                    stdout=f,
                    check=True,
                )

            aligned_files.append(outfile)

            # Build boolean mask:
            # False -> reference sequence
            # True  -> newly added sequence
            total_sequences = reference_count + unseen_count

            mask = np.zeros(total_sequences, dtype=bool)
            mask[reference_count:] = True

            self.unseen_masks[file_idx] = mask

        return aligned_files

    def _process_files(self):
        for file_idx, (msa_file, label) in enumerate(
            zip(self.msa_files, self.labels)
        ):

            records = list(SeqIO.parse(msa_file, "fasta"))

            fname = os.path.basename(msa_file).replace(".fasta", "")

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

        assert len(lengths) == 1, \
            f"Inconsistent sequence lengths: {lengths}"

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

        for (
            seq_id,
            seq,
            label,
            file_idx,
            fname,
        ) in zip(
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

            msa_dict[file_idx]["ids"].append(seq_id)
            msa_dict[file_idx]["seqs"].append(seq)

        return msa_dict
    
    def getUnseenSequenceMasks(self):
        """
        Returns
        -------
        dict[int, np.ndarray]

        Mapping:
            file_idx -> boolean mask

        Example
        -------
        masks = dataset.getUnseenSequenceMasks()

        embeddings = model(msa)      # (N_sequences, D)

        unseen_embeddings = embeddings[masks[file_idx]]
        """
        return self.unseen_masks


# Label mapping
CLASS_MAP = {
    0: "Barrier forming",
    1: "Cation-channel forming",
    2: "Anion-channel forming",
}
