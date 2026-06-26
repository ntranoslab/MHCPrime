import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureProjectionEncoder(nn.Module):
    """
    Shared amino-acid feature encoder with learned projection.

    Maps token indices -> raw AA feature vectors (from a pre-normalized
    feature_table) -> projected feature vectors via a learned Linear layer.

    Assumptions:
      - `feature_table` is a pandas DataFrame with AA letters/tokens as index
        and feature names as columns.
      - Any per-feature normalization (e.g. z-scoring across amino acids) is
        done *before* passing `feature_table` here.
      - `tokenizer` exposes `idx2token`: {token_idx: aa_str}.
    """

    def __init__(
        self,
        vocab_size: int,
        feature_table,
        feature_names,
        tokenizer=None,
        pad_token_id: int = None,
    ):
        super().__init__()

        if tokenizer is None or not hasattr(tokenizer, "idx2token"):
            raise ValueError(
                "FeatureProjectionEncoder requires a tokenizer with `idx2token`."
            )

        self.vocab_size = vocab_size
        self.feature_names = list(feature_names or [])
        if len(self.feature_names) == 0:
            raise ValueError("FeatureProjectionEncoder requires at least one feature.")

        # Raw feature dimensionality (from table)
        self.raw_feature_dim = len(self.feature_names)
        # Projected feature dimensionality (kept equal to raw_dim so it matches
        # the reserved feature block size in SequenceEmbedding_V2)
        self.feature_dim = self.raw_feature_dim

        # Build (vocab_size, raw_feature_dim) feature matrix
        feature_matrix = torch.zeros(
            vocab_size, self.raw_feature_dim, dtype=torch.float32
        )

        missing_aa = set()
        for token_idx, aa in tokenizer.idx2token.items():
            if aa in feature_table.index:
                # values = feature_table.loc[aa, self.feature_names].values
                values = feature_table.loc[aa, self.feature_names].to_numpy(
                    dtype="float32",
                    copy=True,
                )
                feature_matrix[token_idx] = torch.as_tensor(
                    values, dtype=torch.float32
                )
            else:
                missing_aa.add(aa)

        # Optionally force PAD features to zero (hygiene; behavior matches your
        # previous implicit-zero behavior if PAD wasn't in the table)
        if pad_token_id is not None and 0 <= pad_token_id < vocab_size:
            feature_matrix[pad_token_id] = 0.0

        # Store the raw feature matrix as buffer
        self.register_buffer("feature_matrix", feature_matrix)

        if missing_aa:
            print(
                f"[FeatureProjectionEncoder] Warning: no features found for tokens: {sorted(missing_aa)}"
            )

        # Learned projection: raw_feature_dim -> feature_dim (same size, but mixed)
        # This will be xavier-initialized by your global _init_weights (nn.Linear branch).
        self.proj = nn.Linear(self.raw_feature_dim, self.feature_dim, bias=True)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            token_ids: LongTensor of shape (...), e.g. (B, L) or (B*S, L).

        Returns:
            Projected feature tensor of shape (..., feature_dim).
        """
        # Lookup raw features: (..., raw_feature_dim)
        raw_feats = self.feature_matrix[token_ids]

        # Learned projection: (..., feature_dim)
        proj_feats = self.proj(raw_feats)

        return proj_feats


class ContextVectorGenerator(nn.Module):
    """
    Builds a sequence-level context vector from token embeddings using
    content + learned positional information and a lightweight attention
    pooling scheme.

    This is channel-specific (one instance per SequenceEmbedding_V3).
    """

    def __init__(self, embed_dim: int, max_seq_len: int, pad_token_id: int = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # Learned positional embeddings used ONLY for gating context
        # Shape: [max_seq_len, embed_dim]
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Projections for attention pooling
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_emb: torch.Tensor,
                position_ids: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            token_emb:   [B, L, D] token embeddings (no features, no sin/cos)
            position_ids:[B, L] integer positions (0..L-1)
            pad_mask:    [B, L] bool, True for PAD; can be None

        Returns:
            context_vec: [B, D] sequence-level context vector
        """
        B, L, D = token_emb.shape
        assert D == self.embed_dim, "token_emb dim must match embed_dim"

        # Add learned position embeddings
        # pos_emb: [B, L, D]
        pos_emb = self.pos_embed(position_ids.clamp(min=0, max=self.max_seq_len - 1))
        contextual = token_emb + pos_emb  # [B, L, D]

        # Project to Q/K spaces
        Q = self.query_proj(contextual)  # [B, L, D]
        K = self.key_proj(contextual)    # [B, L, D]

        # Build a single global query per sequence by averaging Q over non-pad positions
        if pad_mask is not None:
            # pad_mask: True for PAD -> we zero them out
            valid = (~pad_mask).unsqueeze(-1).float()  # [B, L, 1]
            Q_valid = Q * valid
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1, 1]
            global_q = Q_valid.sum(dim=1, keepdim=True) / denom    # [B, 1, D]
        else:
            global_q = Q.mean(dim=1, keepdim=True)                 # [B, 1, D]

        # Attention scores: [B, 1, L]
        scores = torch.matmul(global_q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        # Mask out PAD positions so they don't get weight
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # [B, 1, L]

        # Weighted sum over contextualized tokens
        context_vec = torch.matmul(attn, contextual).squeeze(1)  # [B, D]
        return context_vec

class FeatureGateMLP(nn.Module):
    """
    Maps a sequence-level context vector to a per-feature gate vector.

    We add a learnable scalar gate_scale (init 0) to ensure that, at
    initialization, the effective gate is ~0 even if outer weight init
    overwrites Linear weights.
    """

    def __init__(self, context_dim: int, feature_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = context_dim

        self.fc1 = nn.Linear(context_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)

        # Multiplicative scale so gate starts at 0 regardless of Linear init
        self.gate_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, context_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_vec: [B, context_dim]

        Returns:
            gate: [B, feature_dim] (can be positive or negative)
        """
        h = F.relu(self.fc1(context_vec))          # [B, hidden_dim]
        raw_gate = self.fc2(h)                     # [B, feature_dim]
        gate = self.gate_scale * raw_gate          # start at 0, then learn scale
        return gate

class SequenceEmbedding_V3(nn.Module):
    """
    Sequence embedding module with:
      - Learnable token embeddings
      - Shared feature projection encoder (FeatureProjectionEncoder)
      - Channel-specific, context-aware gating of features
      - Sin/cos positional encodings added to full [token+feature] embedding

    The forward signature and output shapes are compatible with your previous
    SequenceEmbedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_seq_len: int,
        start_pos: int = 0,
        feature_mode: str = "concat",
        feature_encoder: nn.Module = None,
        use_feature_layernorm: bool = False,
        tokenizer=None,
        pad_token_id: int = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.start_pos = start_pos
        self.feature_mode = feature_mode
        self.feature_encoder = feature_encoder
        self.use_feature_layernorm = use_feature_layernorm
        self.pad_token_id = pad_token_id

        # Only concat mode is supported for now
        if self.feature_mode != "concat":
            raise ValueError(f"SequenceEmbedding_V3 currently supports only 'concat' feature_mode, got {feature_mode}")

        # Determine feature_dim from feature_encoder
        feature_dim = 0
        if self.feature_encoder is not None:
            # Prefer an explicit attribute if present
            if hasattr(self.feature_encoder, "proj_dim"):
                feature_dim = int(self.feature_encoder.proj_dim)
            elif hasattr(self.feature_encoder, "proj") and hasattr(self.feature_encoder.proj, "out_features"):
                feature_dim = int(self.feature_encoder.proj.out_features)
            else:
                raise ValueError(
                    "feature_encoder provided but feature_dim could not be inferred. "
                    "Expected attribute 'proj_dim' or 'proj.out_features'."
                )

        self.feature_dim = feature_dim

        # Token embedding dimension is the remainder
        self.token_dim = self.embedding_dim - self.feature_dim
        if self.token_dim <= 0:
            raise ValueError(
                f"Embedding dim {self.embedding_dim} too small for feature_dim {self.feature_dim}"
            )

        # Learnable token embeddings (channel-specific)
        self.token_embeddings = nn.Embedding(vocab_size, self.token_dim)

        # Optional LayerNorm over projected features
        if self.feature_dim > 0 and self.use_feature_layernorm:
            self.feature_ln = nn.LayerNorm(self.feature_dim)
        else:
            self.feature_ln = None

        # Context generator and gating MLP (channel-specific)
        if self.feature_dim > 0:
            self.context_generator = ContextVectorGenerator(
                embed_dim=self.token_dim,
                max_seq_len=self.max_seq_len,
                pad_token_id=self.pad_token_id,
            )
            self.feature_gate_mlp = FeatureGateMLP(
                context_dim=self.token_dim,
                feature_dim=self.feature_dim,
                hidden_dim=self.token_dim,
            )
        else:
            self.context_generator = None
            self.feature_gate_mlp = None

    @staticmethod
    def _create_positional_encodings(seq_len: int, d_model: int, start_position: int = 0) -> torch.Tensor:
        """
        Standard sin/cos positional encodings, starting from a given position.

        Returns:
            [1, seq_len, d_model]
        """
        position = torch.arange(start_position, start_position + seq_len).unsqueeze(1).float()  # [L, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # [1, L, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token indices
               - [L]
               - [B, L]
               - [B, N, L] (e.g. multiple alleles)

        Returns:
            embeddings:
               - [1, L, D] if input [L]
               - [B, L, D] if input [B, L]
               - [B, N, L, D] if input [B, N, L]
            where D = embedding_dim
        """
        original_shape = x.shape

        # Normalize input to [B_flat, L]
        if len(original_shape) == 1:        # [L]
            B_flat, L = 1, original_shape[0]
            x = x.unsqueeze(0)
            num_alleles = None
        elif len(original_shape) == 2:      # [B, L]
            B_flat, L = original_shape
            num_alleles = None
        elif len(original_shape) == 3:      # [B, N, L]
            B, N, L = original_shape
            B_flat = B * N
            num_alleles = N
            x = x.reshape(B_flat, L)
        else:
            raise ValueError(f"Unexpected input shape for SequenceEmbedding_V3: {original_shape}")

        device = x.device

        # Token embeddings: [B_flat, L, token_dim]
        token_emb = self.token_embeddings(x)

        # Optional feature path
        if self.feature_dim > 0 and self.feature_encoder is not None:
            # Projected features from shared encoder: [B_flat, L, feature_dim]
            features = self.feature_encoder(x)

            if self.feature_ln is not None:
                features = self.feature_ln(features)

            # Build pad mask: True for PAD, else False
            if self.pad_token_id is not None:
                pad_mask = (x == self.pad_token_id)   # [B_flat, L]
            else:
                pad_mask = None

            # Position ids: 0..L-1, independent of start_pos (start_pos used only for PE)
            position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B_flat, -1)  # [B_flat, L]

            # Sequence-level context vector: [B_flat, token_dim]
            context_vec = self.context_generator(
                token_emb=token_emb,
                position_ids=position_ids,
                pad_mask=pad_mask,
            )

            # Context-dependent gate: [B_flat, feature_dim]
            gate = self.feature_gate_mlp(context_vec)  # can be ±

            # Apply gate to features (broadcast over positions)
            gated_features = features * gate.unsqueeze(1)  # [B_flat, L, feature_dim]

            # Concatenate token + gated features
            combined = torch.cat([token_emb, gated_features], dim=-1)  # [B_flat, L, embedding_dim]
        else:
            # No feature path: just token embeddings (should rarely be used in your setup)
            combined = token_emb
            if combined.size(-1) != self.embedding_dim:
                raise ValueError(
                    f"No feature_encoder given but embedding_dim={self.embedding_dim} "
                    f"!= token_dim={self.token_dim}"
                )

        # Add sin/cos positional encodings over FULL embedding_dim
        pos_enc = self._create_positional_encodings(
            seq_len=L,
            d_model=self.embedding_dim,
            start_position=self.start_pos,
        ).to(device)  # [1, L, D]

        combined = combined + pos_enc  # [B_flat, L, D]

        # Restore original shape if needed
        if num_alleles is not None:
            # original [B, N, L] -> [B, N, L, D]
            B = original_shape[0]
            combined = combined.view(B, num_alleles, L, self.embedding_dim)

        return combined


# NOTE: legacy version, used only for BOS token now
class SequenceEmbedding(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim=400, 
        max_seq_len=14, 
        start_pos=0, 
        use_feature_table=False,
        feature_table=None,
        feature_names=None,
        feature_mode="replace",  # "replace" or "concat"
        tokenizer=None,  # Add tokenizer parameter
        pos_enc_type="sincos", # sincos, learned
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.start_pos = start_pos  
        self.use_feature_table = use_feature_table
        self.feature_mode = feature_mode
        self.vocab_size = vocab_size
        self.pos_enc_type=pos_enc_type

        self.idx2token = None
        if tokenizer is not None and hasattr(tokenizer, 'idx2token'):
            self.idx2token = tokenizer.idx2token
        
        self.feature_names = feature_names or []
        self.feature_table = feature_table
        self.num_features = len(self.feature_names) if self.feature_names else 0

        if self.pos_enc_type == "learned":
            max_positions = max_seq_len + start_pos
            self.pos_embeddings = nn.Parameter(torch.zeros(max_positions, embedding_dim))
            with torch.no_grad():
                position = torch.arange(0, max_positions).unsqueeze(1).float()
                div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                                    (-torch.log(torch.tensor(10000.0)) / embedding_dim))
                self.pos_embeddings[:, 0::2] = torch.sin(position * div_term)
                self.pos_embeddings[:, 1::2] = torch.cos(position * div_term)

        if self.use_feature_table and self.num_features > 0:
            if feature_mode == "replace":
                self.base_embedding_dim = embedding_dim - self.num_features
                if self.base_embedding_dim <= 0:
                    raise ValueError(f"Embedding dimension {embedding_dim} is too small for {self.num_features} features in replace mode")
            else:  # "concat" mode
                self.base_embedding_dim = embedding_dim
                self.embedding_dim = embedding_dim + self.num_features
        else:
            self.base_embedding_dim = embedding_dim
        
        self.feature_lookup_tensors = {}
        if self.use_feature_table and feature_table is not None and self.num_features > 0 and self.idx2token is not None:
            for feature_name in self.feature_names:
                if feature_name in feature_table.columns:
                    feature_tensor = torch.zeros(vocab_size)
                    
                    for token_idx, aa in self.idx2token.items():
                        if aa in feature_table.index:
                            feature_tensor[token_idx] = feature_table.loc[aa, feature_name]
                    
                    self.register_buffer(f"feature_{feature_name}", feature_tensor)
                    self.feature_lookup_tensors[feature_name] = f"feature_{feature_name}"
            
            if len(self.feature_lookup_tensors) < self.num_features:
                missing = set(self.feature_names) - set(self.feature_lookup_tensors.keys())
                print(f"Warning: The following features were not found in the feature table: {missing}")
                self.num_features = len(self.feature_lookup_tensors)

        self.word_embeddings = nn.Embedding(vocab_size, self.base_embedding_dim)

    def _map_tokens_to_features_fast(self, x):
        original_shape = x.shape
        device = x.device
        
        x_flat = x.view(-1)
        
        feature_tensors = {}
        
        for feature_name, buffer_name in self.feature_lookup_tensors.items():
            feature_lookup = getattr(self, buffer_name)
            
            if feature_lookup.device != device:
                feature_lookup = feature_lookup.to(device)
            
            feature_values = torch.index_select(feature_lookup, 0, x_flat)
            feature_tensors[feature_name] = feature_values.view(*original_shape)
        
        return feature_tensors

    def _create_positional_encodings(self, seq_len, d_model, start_position=0):
        position = torch.arange(start_position, start_position + seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x, add_pos_enc=True):
        original_shape = x.shape
        
        if len(original_shape) == 1:
            batch_size, seq_len = 1, original_shape[0]
            x = x.unsqueeze(0)
        elif len(original_shape) == 2:
            batch_size, seq_len = original_shape
        elif len(original_shape) == 3:
            batch_size, num_alleles, seq_len = original_shape
            x = x.reshape(-1, seq_len)
        else:
            raise ValueError(f"Unexpected input shape: {original_shape}")

        word_embeddings = self.word_embeddings(x)
        
        if self.use_feature_table and self.num_features > 0 and self.feature_lookup_tensors:
            feature_tensors = self._map_tokens_to_features_fast(x)
            
            if self.feature_mode == "replace":
                # Replace last num_features dimensions with feature values
                base_embed = word_embeddings[..., :-self.num_features] if self.base_embedding_dim < self.embedding_dim else word_embeddings
                
                # Concatenate features along the embedding dimension
                feature_values = torch.stack([feature_tensors[name] for name in self.feature_names if name in feature_tensors], dim=-1)
                
                # Combine base embeddings with features
                embeddings = torch.cat([base_embed, feature_values], dim=-1)
            else:  # "concat" mode
                # Keep full word embeddings and concatenate features
                base_embed = word_embeddings
                
                # Concatenate features along the embedding dimension
                feature_values = torch.stack([feature_tensors[name] for name in self.feature_names if name in feature_tensors], dim=-1)
                
                # Combine base embeddings with features
                embeddings = torch.cat([base_embed, feature_values], dim=-1)
        else:
            embeddings = word_embeddings
        
        if add_pos_enc:
            if self.pos_enc_type == "learned":
                pos_embed = self.pos_embeddings[self.start_pos:self.start_pos + seq_len].unsqueeze(0)
                pos_embed = pos_embed.to(x.device)
                embeddings = embeddings + pos_embed
            else:  # "sincos" (default)
                pos_enc = self._create_positional_encodings(
                    seq_len, 
                    embeddings.size(-1),  # Use actual embedding dimension
                    start_position=self.start_pos
                ).to(x.device)
                embeddings = embeddings + pos_enc

        if len(original_shape) == 3:
            embed_dim = embeddings.size(-1)
            embeddings = embeddings.reshape(batch_size, num_alleles, seq_len, embed_dim)

        return embeddings
