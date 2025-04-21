import torch
import torch.nn as nn

class HierarchicalModel(nn.Module):
    """
    Transformer-based model for hierarchical sequence classification.

    Args:
        cat_sizes (List[int]): vocabulary sizes for each categorical feature.
        num_cont (int): number of continuous features.
        dim (int): embedding dimension.
        seq_depth (int): number of Transformer encoder layers.
        seq_heads (int): number of attention heads.
        num_cls (Dict[str, int]): number of classes for each target.
        seq_len (int): maximum sequence length.
    """
    def __init__(
        self,
        cat_sizes,
        num_cont,
        dim=64,
        seq_depth=6,
        seq_heads=16,
        num_cls=None,
        seq_len=50
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        # Embeddings for categorical features
        self.embs = nn.ModuleList([nn.Embedding(vocab_size, dim) for vocab_size in cat_sizes])
        # Projection for continuous features
        self.cont_proj = nn.Linear(num_cont, dim)

        # Classification token and positional encodings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len + 1, dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=seq_heads,
            batch_first=True,
            dim_feedforward=dim * 2
        )
        self.seq_tr = nn.TransformerEncoder(encoder_layer, num_layers=seq_depth)

        # Separate heads for each target
        if num_cls is None:
            raise ValueError("`num_cls` dict must be provided")
        self.heads = nn.ModuleDict({name: nn.Linear(dim, num_cls[name]) for name in num_cls})

    def forward(self, x_cats, x_conts):
        """
        Args:
            x_cats: LongTensor of shape (B, L, num_categorical)
            x_conts: FloatTensor of shape (B, L, num_continuous)
        Returns:
            Dict[str, Tensor]: logits for each target
        """
        B, L, _ = x_cats.size()
        # Flatten for embedding lookup
        cats_flat = x_cats.view(B * L, -1)
        cont_flat = x_conts.view(B * L, -1)

        # Sum of categorical embeddings
        emb_sum = sum(self.embs[i](cats_flat[:, i]) for i in range(cats_flat.size(1)))
        # Continuous projection
        cont_emb = self.cont_proj(cont_flat)
        # Combine and reshape to sequence
        ev_emb = (emb_sum + cont_emb).view(B, L, self.dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, ev_emb], dim=1) + self.pos_enc

        # Transformer encoder
        seq_out = self.seq_tr(seq)[:, 0, :]  # take CLS output

        # Compute logits for each target
        return {name: head(seq_out) for name, head in self.heads.items()}
