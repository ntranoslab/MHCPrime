import torch
import torch.nn as nn

from .conformer_block import ConformerEncoder
from .ffnn_and_activations import (
    AttentionPoolingHead,
    ESMLanguageModelingHead,
    FFNNHead,
    FFNNHead_w_LayerNorm,
)
from .segment_embedder import SegmentEmbedding
from .sequence_embedders import (
    FeatureProjectionEncoder,
    SequenceEmbedding,
    SequenceEmbedding_V3,
)
from .transformer_layer import TransformerLayer
from .utils import clear_all_gpu_memory, get_trainable_parameters

def get_default_model_params(
    tokenizer,
    processed_feature_table,
    feature_names=None,
):
    if feature_names is None:
        feature_names = list(processed_feature_table.columns)

    return {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": 480,
        "fc_hidden_dims": [240, 120],
        "fc_dropout": 0.1,
        "n_heads": 16,
        "max_peptide_len": 34,
        "max_mhc_len": 34,
        "n_peptide_layers": 2,
        "n_mhc_layers": 1,
        "n_concat_layers": 3,
        "add_pos_enc": True,
        "use_segment_embedder": True,
        "dim_feedforward_multiplier": 1,
        "debugging": False,
        "fc_output_dim": 1,
        "pooling": "cls",
        "ffnn_use_layernorm": False,
        "use_feature_table": True,
        "feature_table": processed_feature_table,
        "sequence_embedder_feature_names": feature_names,
        "sequence_embedder_feature_mode": "concat",
        "sequence_embedder_tokenizer": tokenizer,
        "peptide_start_pos": 1,
        "mhc_start_pos": 35,
        "pos_enc_type": "sincos",
        "seed": 42,

        # peptide conformer
        "add_peptide_conformer": True,
        "peptide_conformer_num_layers": 2,
        "peptide_conformer_n_heads": 16,
        "peptide_conformer_conv_kernel": 5,
        "peptide_conformer_dropout": 0.0,

        # mhc conformer
        "add_mhc_conformer": False,
        "mhc_conformer_num_layers": 2,
        "mhc_conformer_n_heads": 16,
        "mhc_conformer_conv_kernel": 5,
        "mhc_conformer_dropout": 0.0,

        # fusion conformer
        "add_fusion_conformer": True,
        "fusion_conformer_num_layers": 1,
        "fusion_conformer_n_heads": 16,
        "fusion_conformer_conv_kernel": 7,
        "fusion_conformer_dropout": 0.0,
        "fusion_conformer_mask_bos": True,
        "ff_pre_post_swiglu": False,

        # legacy init
        "legacy_init_compat": True,
    }

def init_model(model_params, print_params=False, print_check=False, print_mem=False, device="cuda"):
    model=MHCPrime(**model_params)
    model.to(device)

    model.set_head_type("ffnn")

    if print_params:
        base_trainable_params=get_trainable_parameters(model)
        print(f"Number of trainable modules: {len(base_trainable_params)}")
        total_num_tp=sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {total_num_tp}")
        tp_dict=dict(zip(base_trainable_params, model.parameters()))
        print("Model loaded successfully...", end="\n")

    # delete model lm head
    del model.lm_head

    if print_check:
        print("Checking model parameters before optimizer…")
        for name, p in model.named_parameters():
            print(f"{name:60s} requires_grad={p.requires_grad}  is_leaf={p.is_leaf}")

    if print_mem:
        clear_all_gpu_memory()

    return model

class MHCPrime(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 480,
        fc_hidden_dims=[240, 120],
        fc_dropout: float = 0.1,
        n_heads: int = 16,
        max_peptide_len: int = 34,
        max_mhc_len: int = 34,

        n_peptide_layers: int = 2,
        n_mhc_layers: int = 1,
        n_concat_layers: int = 3,

        add_pos_enc: bool = True,
        use_segment_embedder: bool = True,
        dim_feedforward_multiplier: int = 1,
        transformer_dropout: float = 0.0,
        debugging: bool = False,

        pooling: str = "cls",
        fc_output_dim: int = 1,
        ffnn_use_layernorm: bool = False,

        use_feature_table: bool = True,
        feature_table=None,
        sequence_embedder_feature_names=None,
        sequence_embedder_feature_mode: str = "concat",
        sequence_embedder_tokenizer=None,

        peptide_start_pos: int = 1,
        mhc_start_pos: int = 35,
        pos_enc_type: str = "sincos",
        seed: int = 42,

        add_peptide_conformer: bool = True,
        peptide_conformer_num_layers: int = 2,
        peptide_conformer_n_heads: int = 16,
        peptide_conformer_conv_kernel: int = 5,
        peptide_conformer_dropout: float = 0.0,

        add_mhc_conformer: bool = False,
        mhc_conformer_num_layers: int = 2,
        mhc_conformer_n_heads: int = 16,
        mhc_conformer_conv_kernel: int = 5,
        mhc_conformer_dropout: float = 0.0,

        add_fusion_conformer: bool = True,
        fusion_conformer_num_layers: int = 1,
        fusion_conformer_n_heads: int = 16,
        fusion_conformer_conv_kernel: int = 7,
        fusion_conformer_dropout: float = 0.0,
        fusion_conformer_mask_bos: bool = True,

        ff_pre_post_swiglu: bool = False,

        legacy_init_compat: bool = True,
    ):
        super().__init__()

        self.add_pos_enc = add_pos_enc
        self.transformer_dropout = transformer_dropout
        self.debugging = debugging
        self.fc_output_dim = fc_output_dim
        self.use_segment_embedder = use_segment_embedder
        self.pooling = pooling
        self.dim_feedforward_multiplier = dim_feedforward_multiplier
        self.ffnn_use_layernorm = ffnn_use_layernorm

        self.include_lm_head = bool(legacy_init_compat)
        self.head_type = "ffnn"

        self.enable_feature_tracks = False
        self.feature_tokenizers = {}
        self.feature_lm_heads = nn.ModuleDict()

        self.enable_feature_heads = False
        self.feature_names = []
        self.train_feature_heads_only = False
        self.feature_pooling = "attn"

        self.use_feature_table = use_feature_table
        self.feature_table = feature_table
        self.sequence_embedder_feature_names = sequence_embedder_feature_names
        self.sequence_embedder_feature_mode = sequence_embedder_feature_mode
        self.sequence_embedder_tokenizer = sequence_embedder_tokenizer
        self.pos_enc_type = pos_enc_type

        self.add_peptide_conformer = add_peptide_conformer
        self.peptide_conformer_num_layers = peptide_conformer_num_layers
        self.peptide_conformer_n_heads = peptide_conformer_n_heads
        self.peptide_conformer_conv_kernel = peptide_conformer_conv_kernel
        self.peptide_conformer_dropout = peptide_conformer_dropout

        self.add_mhc_conformer = add_mhc_conformer
        self.mhc_conformer_num_layers = mhc_conformer_num_layers
        self.mhc_conformer_n_heads = mhc_conformer_n_heads
        self.mhc_conformer_conv_kernel = mhc_conformer_conv_kernel
        self.mhc_conformer_dropout = mhc_conformer_dropout

        self.add_fusion_conformer = add_fusion_conformer
        self.fusion_conformer_num_layers = fusion_conformer_num_layers
        self.fusion_conformer_n_heads = fusion_conformer_n_heads
        self.fusion_conformer_conv_kernel = fusion_conformer_conv_kernel
        self.fusion_conformer_dropout = fusion_conformer_dropout
        self.fusion_conformer_mask_bos = fusion_conformer_mask_bos

        self.use_multivector_bos = False
        self.num_mv_bos = 4
        self.mv_pooling_fusion = "stream_mean"

        self.use_multihead_ffnn = False
        self.num_ffnn_heads = 1

        self.include_domain_id = False

        self.ff_pre_post_swiglu = ff_pre_post_swiglu

        self.use_sequence_embedder_v2 = False
        self.use_sequence_embedder_v3 = True

        self.n_peptide_layers = n_peptide_layers
        self.n_mhc_layers = n_mhc_layers

        self.add_episcan_ffnn_head = False

        if self.use_feature_table:
            self.feature_projection_encoder = FeatureProjectionEncoder(
                vocab_size=vocab_size,
                feature_table=self.feature_table,
                feature_names=self.sequence_embedder_feature_names,
                tokenizer=self.sequence_embedder_tokenizer,
                pad_token_id=1,
            )
        else:
            self.feature_projection_encoder = None

        self.peptide_embedder = SequenceEmbedding_V3(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            max_seq_len=max_peptide_len,
            start_pos=peptide_start_pos,
            feature_mode="concat",
            feature_encoder=self.feature_projection_encoder,
            use_feature_layernorm=False,
            tokenizer=self.sequence_embedder_tokenizer,
            pad_token_id=1,
        )

        self.mhc_embedder = SequenceEmbedding_V3(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            max_seq_len=max_mhc_len,
            start_pos=mhc_start_pos,
            feature_mode="concat",
            feature_encoder=self.feature_projection_encoder,
            use_feature_layernorm=False,
            tokenizer=self.sequence_embedder_tokenizer,
            pad_token_id=1,
        )

        self.bos_embedder = SequenceEmbedding(
            vocab_size,
            embedding_dim=embed_dim,
            max_seq_len=1,
            start_pos=0,
        )

        if self.use_segment_embedder:
            self.segment_embedder = SegmentEmbedding(embed_dim, random_seed=seed)

        if self.add_peptide_conformer:
            peptide_conformer_n_start_pos_to_mask = 0

            self.peptide_conformer = ConformerEncoder(
                num_layers=self.peptide_conformer_num_layers,
                d_model=embed_dim,
                n_heads=self.peptide_conformer_n_heads,
                conv_kernel=self.peptide_conformer_conv_kernel,
                dropout=self.peptide_conformer_dropout,
                n_start_pos_to_mask=peptide_conformer_n_start_pos_to_mask,
                ff_pre_post_swiglu=self.ff_pre_post_swiglu,
            )

        if self.add_mhc_conformer:
            self.mhc_conformer = ConformerEncoder(
                num_layers=self.mhc_conformer_num_layers,
                d_model=embed_dim,
                n_heads=self.mhc_conformer_n_heads,
                conv_kernel=self.mhc_conformer_conv_kernel,
                dropout=self.mhc_conformer_dropout,
                ff_pre_post_swiglu=self.ff_pre_post_swiglu,
            )

        if self.add_fusion_conformer:
            fusion_conformer_n_start_pos_to_mask = 0

            if self.fusion_conformer_mask_bos:
                fusion_conformer_n_start_pos_to_mask += 1

            self.fusion_conformer = ConformerEncoder(
                num_layers=self.fusion_conformer_num_layers,
                d_model=embed_dim,
                n_heads=self.fusion_conformer_n_heads,
                conv_kernel=self.fusion_conformer_conv_kernel,
                dropout=self.fusion_conformer_dropout,
                mask_bos=self.fusion_conformer_mask_bos,
                n_start_pos_to_mask=fusion_conformer_n_start_pos_to_mask,
                ff_pre_post_swiglu=self.ff_pre_post_swiglu,
            )

        if self.n_peptide_layers == 1:
            self.peptide_transformer = TransformerLayer(
                embed_dim,
                n_heads,
                dff_mp=self.dim_feedforward_multiplier,
                dropout=self.transformer_dropout,
            )
        else:
            self.peptide_transformers = nn.ModuleList(
                [
                    TransformerLayer(
                        embed_dim,
                        n_heads,
                        dff_mp=self.dim_feedforward_multiplier,
                        dropout=self.transformer_dropout,
                    )
                    for _ in range(self.n_peptide_layers)
                ]
            )

        if self.n_mhc_layers == 1:
            self.mhc_transformer = TransformerLayer(
                embed_dim,
                n_heads,
                dff_mp=self.dim_feedforward_multiplier,
                dropout=self.transformer_dropout,
            )
        else:
            self.mhc_transformers = nn.ModuleList(
                [
                    TransformerLayer(
                        embed_dim,
                        n_heads,
                        dff_mp=self.dim_feedforward_multiplier,
                        dropout=self.transformer_dropout,
                    )
                    for _ in range(self.n_mhc_layers)
                ]
            )

        self.concat_transformers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim,
                    n_heads,
                    dff_mp=self.dim_feedforward_multiplier,
                    dropout=self.transformer_dropout,
                )
                for _ in range(n_concat_layers)
            ]
        )

        if self.pooling == "attn":
            self.ffnn_head = AttentionPoolingHead(
                input_dim=embed_dim,
                hidden_dims=fc_hidden_dims,
                dropout=fc_dropout,
                output_dim=self.fc_output_dim,
                attention_dim=640,
                attention_dropout=0.1,
                use_layer_norm=True,
                activation_fn="swiglu",
            )
        else:
            if self.ffnn_use_layernorm:
                self.ffnn_head = FFNNHead_w_LayerNorm(
                    embed_dim,
                    hidden_dims=fc_hidden_dims,
                    dropout=fc_dropout,
                    output_dim=self.fc_output_dim,
                    pooling=self.pooling,
                    use_layernorm=True,
                )
            else:
                self.ffnn_head = FFNNHead(
                    embed_dim,
                    hidden_dims=fc_hidden_dims,
                    dropout=fc_dropout,
                    output_dim=self.fc_output_dim,
                    pooling=self.pooling,
                )

        if legacy_init_compat:
            self.lm_head = ESMLanguageModelingHead(embed_dim, vocab_size)

        self.feature_heads = nn.ModuleDict()

        self.feature_embedders = nn.ModuleDict()

        if legacy_init_compat:
            nn.init.xavier_uniform_(self.lm_head.dense.weight)
            if self.lm_head.dense.bias is not None:
                self.lm_head.dense.bias.data.zero_()
            nn.init.xavier_uniform_(self.lm_head.decoder.weight)
            self.lm_head.layer_norm.weight.data.fill_(1.0)
            self.lm_head.layer_norm.bias.data.zero_()

        self.set_head_type(self.head_type)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def set_head_type(self, head_type):
        if head_type not in ["ffnn", "lm"]:
            raise ValueError(f"Invalid head type: {head_type}. Must be 'ffnn' or 'lm'.")

        if not self.include_lm_head and head_type == "lm":
            raise ValueError("Cannot set head_type to 'lm' when include_lm_head is False.")

        self.head_type = head_type

        if self.include_lm_head:
            for param in self.ffnn_head.parameters():
                param.requires_grad = head_type == "ffnn" or not self.train_feature_heads_only

            for param in self.lm_head.parameters():
                param.requires_grad = head_type == "lm"

        if self.enable_feature_heads:
            for head in self.feature_heads.values():
                for param in head.parameters():
                    param.requires_grad = True

            if self.train_feature_heads_only and head_type == "ffnn":
                for param in self.ffnn_head.parameters():
                    param.requires_grad = False
    def forward(self, batch, return_embeddings=False, output_attentions=False):
        peptide_seq = batch["peptide"]
        peptide_mask = batch["peptide_mask"]
        mhc_list = batch["mhc_list"]
        mhc_mask_list = batch["mhc_mask_list"]

        attention_weights = None
        if output_attentions:
            attention_weights = {
                "peptide": [],
                "mhc": [],
                "concat": [],
            }

        if self.debugging:
            print("RUNNING PEPTIDE TRANSFORMER")
            print(end="\n")

        peptide_emb = self.peptide_embedder(peptide_seq)

        if self.add_peptide_conformer:
            peptide_emb = self.peptide_conformer(peptide_emb, mask=peptide_mask)

        if self.use_segment_embedder:
            peptide_emb = self.segment_embedder(peptide_emb, 0)

        if self.n_peptide_layers == 1:
            if output_attentions:
                peptide_enc, peptide_attn = self.peptide_transformer(
                    peptide_emb,
                    ~peptide_mask.to(peptide_seq.device),
                    output_attentions=True,
                )
                attention_weights["peptide"] = peptide_attn
            else:
                peptide_enc = self.peptide_transformer(
                    peptide_emb,
                    ~peptide_mask.to(peptide_seq.device),
                )
        else:
            peptide_enc = peptide_emb
            for transformer in self.peptide_transformers:
                if output_attentions:
                    peptide_enc, peptide_attn = transformer(
                        peptide_enc,
                        ~peptide_mask.to(peptide_seq.device),
                        output_attentions=True,
                    )
                    attention_weights["peptide"] = peptide_attn
                else:
                    peptide_enc = transformer(
                        peptide_enc,
                        ~peptide_mask.to(peptide_seq.device),
                    )

        bos_token = torch.zeros(
            (peptide_seq.size(0), 1),
            dtype=torch.long,
            device=peptide_seq.device,
        )
        bos_emb = self.bos_embedder(bos_token, add_pos_enc=False)

        if self.fc_output_dim == 1:
            max_logits = torch.full(
                (peptide_seq.size(0),),
                -float("inf"),
                device=peptide_seq.device,
            )
            min_logits = torch.full(
                (peptide_seq.size(0),),
                float("inf"),
                device=peptide_seq.device,
            )
        elif self.fc_output_dim == 2:
            max_logits = torch.full(
                (peptide_seq.size(0), 2),
                -float("inf"),
                device=peptide_seq.device,
            )
            min_logits = torch.full(
                (peptide_seq.size(0), 2),
                float("inf"),
                device=peptide_seq.device,
            )
        else:
            raise ValueError("fc_output_dim must be 1 or 2.")

        max_indices = torch.zeros(
            peptide_seq.size(0),
            dtype=torch.long,
            device=peptide_seq.device,
        )
        min_indices = torch.zeros(
            peptide_seq.size(0),
            dtype=torch.long,
            device=peptide_seq.device,
        )

        all_logits = []
        all_bos_embeddings = []

        if output_attentions:
            all_mhc_attentions = []
            all_concat_attentions = []

        for i in range(mhc_list.shape[1]):
            mhc_seq = mhc_list[:, i, :]
            mhc_mask = mhc_mask_list[:, i, :]

            mhc_emb = self.mhc_embedder(mhc_seq)

            if self.add_mhc_conformer:
                mhc_emb = self.mhc_conformer(mhc_emb, mask=mhc_mask)

            if self.use_segment_embedder:
                mhc_emb = self.segment_embedder(mhc_emb, 1)

            if self.debugging:
                print("RUNNING MHC TRANSFORMER")
                print(end="\n")

            if self.n_mhc_layers == 1:
                if output_attentions:
                    mhc_enc, mhc_attn = self.mhc_transformer(
                        mhc_emb,
                        ~mhc_mask,
                        output_attentions=True,
                    )
                    all_mhc_attentions.append(mhc_attn)
                else:
                    mhc_enc = self.mhc_transformer(mhc_emb, ~mhc_mask)
            else:
                mhc_enc = mhc_emb
                for transformer in self.mhc_transformers:
                    if output_attentions:
                        mhc_enc, mhc_attn = transformer(
                            mhc_enc,
                            ~mhc_mask,
                            output_attentions=True,
                        )
                        all_mhc_attentions.append(mhc_attn)
                    else:
                        mhc_enc = transformer(mhc_enc, ~mhc_mask)

            concat_seq = torch.cat([bos_emb, peptide_enc, mhc_enc], dim=1)
            concat_mask = torch.cat(
                [
                    torch.ones(
                        (peptide_seq.size(0), 1),
                        dtype=bool,
                        device=peptide_seq.device,
                    ),
                    peptide_mask,
                    mhc_mask,
                ],
                dim=1,
            )

            concat_layer_attentions = []

            for j, transformer in enumerate(self.concat_transformers):
                if output_attentions:
                    concat_seq, concat_attn = transformer(
                        concat_seq,
                        ~concat_mask,
                        output_attentions=True,
                    )
                    concat_layer_attentions.append(concat_attn)
                else:
                    if self.debugging:
                        print(f"RUNNING CONCAT TRANSFORMER LAYER {j}")
                        print(end="\n")

                    concat_seq = transformer(concat_seq, ~concat_mask)

            if self.add_fusion_conformer:
                concat_seq = self.fusion_conformer(concat_seq, mask=concat_mask)

            if output_attentions:
                all_concat_attentions.append(concat_layer_attentions)

            bos_embedding = concat_seq[:, 0, :]
            all_bos_embeddings.append(bos_embedding)

            if self.fc_output_dim == 1:
                if self.pooling == "attn":
                    logits = self.ffnn_head(concat_seq, ~concat_mask).squeeze(-1)
                else:
                    logits = self.ffnn_head(concat_seq).squeeze(-1)

            elif self.fc_output_dim == 2:
                if self.pooling == "attn":
                    logits = self.ffnn_head(concat_seq, ~concat_mask)
                else:
                    logits = self.ffnn_head(concat_seq)

            all_logits.append(logits)

            if self.fc_output_dim == 1:
                update_mask = logits > max_logits
                max_logits = torch.where(update_mask, logits, max_logits)

                update_mask_neg = logits < min_logits
                min_logits = torch.where(update_mask_neg, logits, min_logits)

            else:
                update_mask = logits[:, 1] > max_logits[:, 1]
                max_logits = torch.where(update_mask.unsqueeze(-1), logits, max_logits)

                update_mask_neg = logits[:, 1] < min_logits[:, 1]
                min_logits = torch.where(
                    update_mask_neg.unsqueeze(-1),
                    logits,
                    min_logits,
                )

            max_indices = torch.where(update_mask, i, max_indices)
            min_indices = torch.where(update_mask_neg, i, min_indices)

        all_logits = torch.stack(all_logits, dim=1)
        all_bos_embeddings = torch.stack(all_bos_embeddings, dim=1)

        if "label" in batch:
            if self.fc_output_dim == 1:
                selected_logits = torch.where(
                    batch["label"] == 1,
                    all_logits[torch.arange(len(max_indices)), max_indices],
                    all_logits[torch.arange(len(min_indices)), min_indices],
                )
            else:
                selected_logits = torch.where(
                    batch["label"].unsqueeze(-1) == 1,
                    all_logits[torch.arange(len(max_indices)), max_indices, :],
                    all_logits[torch.arange(len(min_indices)), min_indices, :],
                )

            selected_indices = torch.where(
                batch["label"] == 1,
                max_indices,
                min_indices,
            )

        else:
            if self.fc_output_dim == 1:
                selected_logits = all_logits[
                    torch.arange(len(max_indices)),
                    max_indices,
                ]
            else:
                selected_logits = all_logits[
                    torch.arange(len(max_indices)),
                    max_indices,
                    :,
                ]

            selected_indices = max_indices

        if output_attentions:
            selected_mhc_attentions = []
            selected_concat_attentions = []

            for b in range(len(selected_indices)):
                selected_idx = selected_indices[b].item()
                selected_mhc_attentions.append(all_mhc_attentions[selected_idx][b])

                selected_concat_layers = []
                for layer_idx in range(len(all_concat_attentions[selected_idx])):
                    selected_concat_layers.append(
                        all_concat_attentions[selected_idx][layer_idx][b]
                    )
                selected_concat_attentions.append(selected_concat_layers)

            attention_weights["mhc"] = selected_mhc_attentions
            attention_weights["concat"] = selected_concat_attentions

        results = [selected_logits]

        if return_embeddings:
            selected_bos_embeddings = all_bos_embeddings[
                torch.arange(len(selected_indices)),
                selected_indices,
            ]
            results.append(selected_bos_embeddings)

        if output_attentions:
            results.append(attention_weights)

        return tuple(results) if len(results) > 1 else results[0]