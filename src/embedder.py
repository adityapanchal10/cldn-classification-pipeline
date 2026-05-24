import esm
import torch

def load_msa_transformer(model_name="esm_msa1b_t12_100M_UR50S"):
    """Load a pretrained ESM MSA Transformer model and its alphabet."""
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    return model, alphabet

class MSAEmbedder:
    """
    Handles cleaning, padding/truncation, and embedding of MSA sequences.

    Two embedding modes:
      - embed_msa()                    : true MSA-mode (all seqs from one file together)
      - embed_sequences_per_residue()  : fallback for individual sequence inference
    """

    def __init__(self, model_name="esm_msa1b_t12_100M_UR50S", device=None):
        self.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device is None else device
        )
        self.model, self.alphabet = load_msa_transformer(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model           = self.model.to(self.device)
        self.valid_chars     = set(self.alphabet.all_toks)
        self.model.eval()
        print(f'\nUsing model: {self.model.__class__.__name__} ({model_name})')

    def _clean_sequence(self, seq: str) -> str:
        """Replace invalid characters with gap characters."""
        return "".join(c if c in self.valid_chars else '-' for c in seq).upper()

    @staticmethod
    def pad_or_truncate(sequences, seq_length, pad_char='-'):
        """Ensure all sequences have exactly seq_length residues."""
        return [
            seq[:seq_length] if len(seq) > seq_length else seq.ljust(seq_length, pad_char)
            for seq in sequences
        ]

    def embed_msa(self, sequences, seq_length=190, max_msa_depth=600):
        """
        Embed all sequences from ONE MSA file together (true MSA mode).
        Column attention operates across all sequences simultaneously.

        Args:
            sequences    : list of aligned sequences (all from the same MSA file)
            seq_length   : pad/truncate target length
            max_msa_depth: max sequences per forward pass (GPU memory limit)

        Returns:
            Tensor of shape (N, seq_length, 768)
        """
        sequences = [self._clean_sequence(s) for s in sequences]
        sequences = self.pad_or_truncate(sequences, seq_length)
        N = len(sequences)

        all_embeddings = []

        for start in range(0, N, max_msa_depth):
            chunk = sequences[start: start + max_msa_depth]

            # Wrap all chunk sequences as a single MSA input
            msa_input = [(f'seq{start + j}', seq) for j, seq in enumerate(chunk)]

            # batch_converter: tokens shape → (1, depth, seq_len+1), +1 for BOS
            _, _, batch_tokens = self.batch_converter([msa_input])
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[12], return_contacts=False)

            # Extract representations: (1, depth, seq_len+1, 768)
            token_emb = results["representations"][12]
            token_emb = token_emb[:, :, 1:, :]    # Remove BOS → (1, depth, seq_len, 768)
            token_emb = token_emb.squeeze(0)       # → (depth, seq_len, 768)

            all_embeddings.append(token_emb.cpu())

        output_embeddings = torch.cat(all_embeddings, dim=0)
        assert len(output_embeddings.shape) == 3, f"Unexpected shape: {output_embeddings.shape}"
        return output_embeddings  # (N, seq_len, 768)

    def embed_sequences_per_residue(
        self,
        sequences,
        seq_length=190,
        batch_size=32
    ):
        """
        Embed sequences independently.

        Returns:
            Tensor: (N, L, 768)
        """

        sequences = [self._clean_sequence(s) for s in sequences]
        sequences = self.pad_or_truncate(sequences, seq_length)

        all_embeddings = []

        total_batches = (
            len(sequences) + batch_size - 1
        ) // batch_size

        for batch_idx in range(total_batches):

            start = batch_idx * batch_size
            end   = min(start + batch_size, len(sequences))

            batch = sequences[start:end]

            msa_inputs = [
                [(f"seq{i}", seq)]
                for i, seq in enumerate(batch)
            ]

            _, _, batch_tokens = self.batch_converter(msa_inputs)

            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():

                results = self.model(
                    batch_tokens,
                    repr_layers=[12],
                    return_contacts=False
                )

            token_emb = results["representations"][12]

            # Shape:
            # (B, 1, L+1, 768)

            token_emb = token_emb[:, 0, 1:, :]

            all_embeddings.append(token_emb.cpu())

        output_embeddings = torch.cat(all_embeddings, dim=0)

        return output_embeddings