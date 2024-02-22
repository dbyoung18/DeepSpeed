# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from ..common_parameters import *
from ..layer_container_base import LayerContainer
'''
 # HF GPTJ model looks like this:

GPTJForCausalLM(
  (transformer): GPTJModel(
    (wte): Embedding(50400, 4096)
    (drop): Dropout(p=0.0, inplace=False)
    (h): ModuleList(
      (0-27): 28 x GPTJBlock(
        (ln_1): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (attn): GPTJAttention(
          (position_emb): GPTJRotaryEmbedding()
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (out_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (residual_drop): Dropout(p=0.0, inplace=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): GPTJMLP(
          (fc_in): Linear(in_features=4096, out_features=16384, bias=True)
          (fc_out): Linear(in_features=16384, out_features=4096, bias=True)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=4096, out_features=50400, bias=True)
)
'''


class GPTJTransformerContainer(LayerContainer):
    """
        Transformer layer container for the GPT-J model.
    """
    ln_attn_gamma: NormParameter
    ln_attn_beta: NormParameter

    qkv_w: UnfusedQKVParameter
    attn_out_w: AttentionOutputParameter

    mlp_1_w: MLP1Parameter
    mlp_1_b: MLP1Parameter
    mlp_2_w: MLP2Parameter
    mlp_2_b: MLP2Parameter

    PARAM_MAPPING = {
      # Layer Norm
      "ln_1.weight": "ln_attn_gamma.params",
      "ln_1.bias": "ln_attn_beta.params",
      # Masked MHA
      "attn.q_proj.weight": "qkv_w.q_params",
      "attn.k_proj.weight": "qkv_w.k_params",
      "attn.v_proj.weight": "qkv_w.v_params",
      "attn.out_proj.weight": "attn_out_w.params",
      # "attn.masked_bias": "pytorch_model-00001-of-00003.bin",
      # "attn.bias": "pytorch_model-00001-of-00003.bin",
      # Feed Forward
      "mlp.fc_in.weight": "mlp_1_w.params",
      "mlp.fc_in.bias": "mlp_1_b.params",
      "mlp.fc_out.weight": "mlp_2_w.params",
      "mlp.fc_out.bias": "mlp_2_b.params",
    }


class GPTJNonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the GPT-J model.
    """
    word_emb: EmbeddingParameter
    word_unembed_w: UnembedParameter
    word_unembed_b: UnembedParameter
    final_norm_gamma: NormParameter
    final_norm_beta: NormParameter

    PARAM_MAPPING = {
        # Input Embedding
        "transformer.wte.weight": "word_emb.params",
        # Layer Norm
        "transformer.ln_f.weight": "final_norm_gamma.params",
        "transformer.ln_f.bias": "final_norm_beta.params",
        # Linear
        "lm_head.weight": "word_unembed_w.params",
        "lm_head.bias": "word_unembed_b.params"
    }
