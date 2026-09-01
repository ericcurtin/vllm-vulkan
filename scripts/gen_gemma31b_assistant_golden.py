#!/usr/bin/env python3
"""Generate the INC-3 golden fixture for the Gemma4-31B EAGLE drafter
(gemma4_assistant) CPU forward.

Loads the REAL bf16 drafter checkpoint
(~/repos/OminiX-MLX/models/gemma-4-31B-it-assistant/model.safetensors) and
drives it through the upstream mlx_vlm `Attention`/`DecoderLayer` classes
(gemma4/language.py) directly — the same modules the shipping mlx_vlm
gemma4_assistant drafter uses internally — wired exactly like
`Gemma4AssistantDraftModel.__call__` / `draft_block`
(mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py) and the
OminiX-MLX Rust port (gemma4-mlx/src/assistant.rs):

  1. inputs_embeds = concat([prev_token_embed, recurrent_hidden], axis=-1)   (embed FIRST)
  2. h = pre_projection(inputs_embeds)
  3. for each of the 4 decoder layers (kv_shared_only=True, no k/v_proj):
       shared_kv = borrowed (K, V) of the matching layer_type, scale=1.0,
       own Q rotated via the layer's own RoPE (proportional on the global
       layer, default on sliding), q_norm'd; borrowed K NOT re-rotated.
  4. inner = model.norm(h)
  5. post_projection_out = post_projection(inner)     # next recurrent_hidden
  6. logits = embed_tokens.as_linear(inner)            # tied lm_head, no softcap

The "tiny fixed input" (prev_token_embed, recurrent_hidden, borrowed K/V) is
synthetic (deterministic PRNG) — only the WEIGHTS are the real checkpoint.
This exercises exactly the load-bearing knobs called out in
scripts/GEMMA31B_SPEC_PLAN.md section 5 risk #5: embed-first concat order,
scale=1.0 (not 1/sqrt(head_dim)), and the KV-borrow (no re-rotation of K).

Run with the mlx_vlm venv (system python3 lacks mlx_vlm):
  VLLM_TEST_GEMMA31B_ASSISTANT_DIR=<checkpoint dir> \
      <mlx_vlm venv python3> scripts/gen_gemma31b_assistant_golden.py [out_dir]

<out_dir> defaults to tests/fixtures/gemma31b_assistant_golden/, the directory
`src/gemma_assistant.rs`'s golden test reads.

Writes flat little-endian float32 .npy files to <out_dir>:
  prev_token_embed.npy [5376]        recurrent_hidden.npy [5376]
  sliding_k.npy [4,16,256]           sliding_v.npy [4,16,256]
  full_k.npy    [4,4,512]            full_v.npy    [4,4,512]
  logits.npy    [262144]             post_proj.npy [5376]
"""

import os
import sys

import mlx.core as mx
import numpy as np
from mlx_vlm.models.gemma4.config import TextConfig
from mlx_vlm.models.gemma4.language import DecoderLayer

# Drafter checkpoint directory, from the environment — the SAME variable the
# Rust golden test (`src/gemma_assistant.rs`) reads, so a fixture regenerated
# here and the test that consumes it always point at one checkpoint. There is
# deliberately no hard-coded default: a machine-specific absolute path is how
# these fixtures became unreproducible in the first place.
_DIR = os.environ.get("VLLM_TEST_GEMMA31B_ASSISTANT_DIR") or os.environ.get(
    "GEMMA31B_ASSISTANT_DIR"
)
if not _DIR:
    sys.exit(
        "set VLLM_TEST_GEMMA31B_ASSISTANT_DIR to the gemma-4-31B-it-assistant checkpoint directory"
    )
CKPT = os.path.join(_DIR, "model.safetensors")

HIDDEN = 1024
BACKBONE = 5376
VOCAB = 262144
KV_LEN = 4
POSITION_OFFSET = 7  # arbitrary fixed offset; constant across the (unused-here) K-loop


def save_npy(path, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    np.save(path, arr)


def rng_array(rng, shape, scale=1.0):
    return rng.standard_normal(shape).astype(np.float32) * scale


def main(out_dir: str):
    weights = {k: v.astype(mx.float32) for k, v in mx.load(CKPT).items()}

    cfg = TextConfig(
        model_type="gemma4_text",
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        intermediate_size=8192,
        num_attention_heads=32,
        head_dim=256,
        global_head_dim=512,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        num_kv_shared_layers=0,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        hidden_activation="gelu_pytorch_tanh",
        rope_traditional=False,
        sliding_window=1024,
        attention_bias=False,
        attention_k_eq_v=True,
        final_logit_softcapping=None,
        use_double_wide_mlp=False,
        enable_moe_block=False,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        tie_word_embeddings=True,
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        },
    )

    layers = [DecoderLayer(cfg, layer_idx=i, kv_shared_only=True) for i in range(4)]
    for i, layer in enumerate(layers):
        p = f"model.layers.{i}"
        layer.input_layernorm.weight = weights[f"{p}.input_layernorm.weight"]
        layer.post_attention_layernorm.weight = weights[
            f"{p}.post_attention_layernorm.weight"
        ]
        layer.pre_feedforward_layernorm.weight = weights[
            f"{p}.pre_feedforward_layernorm.weight"
        ]
        layer.post_feedforward_layernorm.weight = weights[
            f"{p}.post_feedforward_layernorm.weight"
        ]
        layer.layer_scalar = weights[f"{p}.layer_scalar"]
        layer.mlp.gate_proj.weight = weights[f"{p}.mlp.gate_proj.weight"]
        layer.mlp.up_proj.weight = weights[f"{p}.mlp.up_proj.weight"]
        layer.mlp.down_proj.weight = weights[f"{p}.mlp.down_proj.weight"]
        sa = layer.self_attn
        sa.q_proj.weight = weights[f"{p}.self_attn.q_proj.weight"]
        sa.o_proj.weight = weights[f"{p}.self_attn.o_proj.weight"]
        sa.q_norm.weight = weights[f"{p}.self_attn.q_norm.weight"]

    norm_w = weights["model.norm.weight"]
    embed_w = weights["model.embed_tokens.weight"]  # [VOCAB, HIDDEN], tied lm_head
    pre_proj_w = weights["pre_projection.weight"]  # [HIDDEN, 2*BACKBONE]
    post_proj_w = weights["post_projection.weight"]  # [BACKBONE, HIDDEN]

    # ---- synthetic tiny fixed input (deterministic) ----
    rng = np.random.default_rng(1234)
    prev_token_embed = rng_array(rng, (BACKBONE,), scale=0.05)
    recurrent_hidden = rng_array(rng, (BACKBONE,), scale=0.05)
    sliding_k = rng_array(rng, (KV_LEN, 16, 256), scale=0.05)
    sliding_v = rng_array(rng, (KV_LEN, 16, 256), scale=0.05)
    full_k = rng_array(rng, (KV_LEN, 4, 512), scale=0.05)
    full_v = rng_array(rng, (KV_LEN, 4, 512), scale=0.05)

    # ---- forward ----
    inputs_embeds = mx.concatenate(
        [
            mx.array(prev_token_embed)[None, None, :],
            mx.array(recurrent_hidden)[None, None, :],
        ],
        axis=-1,
    )  # [1,1,2*BACKBONE], embed FIRST
    h = inputs_embeds @ pre_proj_w.T  # pre_projection, bias=False

    sk_full = mx.array(full_k)[None].transpose(
        0, 2, 1, 3
    )  # [1, n_kv, kv_len, hd] to match KVCache-style layer input
    sv_full = mx.array(full_v)[None].transpose(0, 2, 1, 3)
    sk_slide = mx.array(sliding_k)[None].transpose(0, 2, 1, 3)
    sv_slide = mx.array(sliding_v)[None].transpose(0, 2, 1, 3)

    offset = mx.array(POSITION_OFFSET)
    for layer in layers:
        shared_kv = (
            (sk_slide, sv_slide)
            if layer.layer_type == "sliding_attention"
            else (sk_full, sv_full)
        )
        h, _, _ = layer(
            h,
            mask=None,
            cache=None,
            per_layer_input=None,
            shared_kv=shared_kv,
            offset=offset,
        )

    inner = mx.fast.rms_norm(h, norm_w, cfg.rms_norm_eps)
    post_proj_out = inner @ post_proj_w.T  # post_projection, bias=False
    logits = inner @ embed_w.T  # tied lm_head, NO softcap

    mx.eval(logits, post_proj_out)

    logits_np = np.array(logits.reshape(-1).astype(mx.float32))
    post_proj_np = np.array(post_proj_out.reshape(-1).astype(mx.float32))

    save_npy(f"{out_dir}/prev_token_embed.npy", prev_token_embed)
    save_npy(f"{out_dir}/recurrent_hidden.npy", recurrent_hidden)
    save_npy(f"{out_dir}/sliding_k.npy", sliding_k)
    save_npy(f"{out_dir}/sliding_v.npy", sliding_v)
    save_npy(f"{out_dir}/full_k.npy", full_k)
    save_npy(f"{out_dir}/full_v.npy", full_v)
    save_npy(f"{out_dir}/logits.npy", logits_np)
    save_npy(f"{out_dir}/post_proj.npy", post_proj_np)

    print(f"wrote golden fixtures to {out_dir}")
    print(f"logits argmax={int(np.argmax(logits_np))} val={logits_np.max():.4f}")
    print(f"post_proj[:5]={post_proj_np[:5]}")


if __name__ == "__main__":
    # Default to the in-repo fixture directory the Rust test reads, so a bare
    # run regenerates exactly the committed fixtures.
    default_out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests",
        "fixtures",
        "gemma31b_assistant_golden",
    )
    out = sys.argv[1] if len(sys.argv) > 1 else default_out
    os.makedirs(out, exist_ok=True)
    main(out)
