# Gemma4-31B EAGLE-drafter Speculative Decoding — Mac-side Prep Plan

Branch `perf/gemma-31b-spec` (off `perf/gemma-12b`). Worktree
`~/repos/vllm-vulkan-gemma31b`. Target box: BC-250 cluster, TP-4, RADV GFX1013.
All work LOCAL COMMITS ONLY.

This doc resolves the EAGLE-Gemma4 wiring from the authoritative references,
pins the `g31b()` config, and hands the fast-worker an ordered increment list
with per-increment Mac-CPU validation gates.

## Authoritative references (cited, do not re-derive)
- HF transformers 5.7.0.dev0 `Gemma4AssistantForCausalLM`:
  `/opt/homebrew/lib/python3.14/site-packages/transformers/models/gemma4_assistant/modeling_gemma4_assistant.py`
  (structure: pre/post_projection, forward, shared_kv_states).
- MLX drafter (the **caller** — draft loop + pre_projection assembly):
  `~/.cache/uv/.../mlx_vlm/speculative/drafters/gemma4_assistant/gemma4_assistant.py`
  (`Gemma4AssistantDraftModel.__call__` + `draft_block`).
- OminiX-MLX Rust port (our closest prior art, near-drop-in semantics):
  `~/repos/OminiX-MLX/gemma4-mlx/src/assistant.rs` (attention scale, sliding-mask,
  concat order, KV-borrow, loader tensor map).
- Qwen verify/rollback pattern to adapt (do NOT depend on it):
  `~/repos/vllm-vulkan-qwenspec/src/qwen35_forward.rs`
  `forward_qwen35_verify_core` (L3051), `qwen35_verify_rollback_impl` (L3216).
- Real configs: `gemma-4-31B-it-NVFP4/config.json`,
  `gemma-4-31B-it-assistant/config.json` (both under `~/repos/OminiX-MLX/models/`).

---

## 1. Resolved EAGLE-Gemma4 semantics (the 5 unknowns)

### (1) `pre_projection` input assembly — `[.,10752] = concat(A[5376], B[5376])`
`pre_projection: Linear(2*backbone_hidden=10752 -> drafter_hidden=1024)`
(modeling L125). The 10752 input is **`concat([prev_token_embed, recurrent_hidden])`
on the last axis, embed FIRST** (gemma4_assistant.py L246-247;
assistant.rs L706-739 `build_inputs_embeds`, with the explicit warning that
recurrent-first collapses acceptance 0.43→0.01).
- **`prev_token_embed` [5376]** = the **TARGET's** `embed_tokens(prev_token) *
  target.embed_scale` (gemma4_assistant.py L89-90, L246). It is the target's
  backbone-space (5376) embedding of the last sampled/bonus token — NOT the
  drafter's own 1024-dim embedding.
- **`recurrent_hidden` [5376]** = on **step 0** the **target's last hidden
  state** (backbone 5376, post-final-decoder, the hidden that produced the bonus
  token); on **step k>0** the drafter's own `post_projection` output from k-1
  (assistant.rs L9-12, gemma4_assistant.py L242-248).

So the two halves are NOT two aux layers — they are (target-embed-of-prev-token)
‖ (rolling target-space hidden). Only ONE target layer feeds this: the target's
FINAL hidden (step 0), thereafter self-recurrent.

### (2) KV-share source — target's last-layer-of-each-type K/V, borrowed as-is
The drafter is **Q-only**: it has `q_proj`, `q_norm`, `o_proj` and **no
k_proj/v_proj/k_norm** (drafter tensor list; assistant.rs L14-20). Its attention
is **bidirectional cross-attention over the target's `shared_kv_states`**, a dict
`{ "full_attention": (K,V), "sliding_attention": (K,V) }` = the **K/V of the
target's LAST layer of each attention type** (modeling docstring L146-147;
assistant.rs L297-306). For the 31B target (period-6 globals):
- drafter sliding layers 0,1,2  ← target **layer 58** (last sliding) K/V.
- drafter global layer 3        ← target **layer 59** (last full) K/V.

Borrowed K/V are consumed **exactly as stored by the target** — already
RoPE-rotated and already normed; the drafter does **not** re-rotate borrowed K
(assistant.rs L18-20, L157-198). The drafter **does** rotate its OWN queries and
applies its own `q_norm` (assistant.rs L168-173).

Head-dim/kv-head reconciliation — **drafter and target share identical
per-type dims, so KV is directly reusable, no reshape**:
| type    | drafter q-heads×hd | drafter kv-heads×hd | target kv-heads×hd |
|---------|--------------------|--------------------|--------------------|
| sliding | 32×256 (=8192 q_proj) | 16×256 | 16×256 |
| global  | 32×512 (=16384 q_proj)| 4×512  | 4×512  |
(drafter config `head_dim 256`, `global_head_dim 512`, `num_key_value_heads 16`,
`num_global_key_value_heads 4` == target's.) GQA groups: sliding 32/16=2,
global 32/4=8 — both integer.

**Attention scale = 1.0** (NOT 1/sqrt(head_dim)) by default — matches the
target's QK-norm softmax convention under which the borrowed K/V were produced
(assistant.rs L598-608; env `MTPLX_PAIR_RSQRT_SCALE=1` A/B toggle). Sliding
layers additionally mask to the last `sliding_window=1024` KV positions when
`kv_len>1024` (assistant.rs L185-198).

### (3) Output path — drafter-internal norm + tied lm_head, NO softcap, NO target
`draft_logits = drafter.embed_tokens^T @ drafter.norm(h)` where `h` is the
drafter's post-layer hidden [1024]. I.e. **the drafter's OWN final RMSNorm
(`model.norm[1024]`) then its OWN tied lm_head over `embed_tokens[262144,1024]`**
(modeling L182-188, gemma4_assistant.py L186-192, assistant.rs L398-409).
- **This CORRECTS the deliverable-3 hypothesis**: draft logits do **NOT** go
  through `post_projection`, the target's `model.norm`, or the target lm_head.
- **NO final-logit softcap** on the draft path (drafter
  `final_logit_softcapping: null`) — unlike the 31B target's 30.0.
- `post_projection: Linear(1024 -> 5376)` is a **separate** output: it produces
  the next step's `recurrent_hidden` (fed back to pre_projection). It is NOT on
  the logit path. It reads the same post-`norm` hidden as the lm_head.

### (4) The draft-K loop (`draft_block`, gemma4_assistant.py L195-252)
Drafts `K = block_size-1` tokens autoregressively:
```
tok      = last_bonus            # target's last accepted/bonus token id
h_prev   = target_last_hidden    # 5376 backbone hidden (step-0 seed)
pos      = fixed_offset          # position of bonus tok; CONSTANT all K steps
shared_kv= FIXED target KV snapshot   # does NOT grow across draft steps
for _ in 0..K:
    tok_embed     = target.embed(tok) * target.embed_scale     # [5376]
    inputs_embeds = concat([tok_embed, h_prev])                # [10752], embed-first
    h_prev, logits = drafter(inputs_embeds, shared_kv, pos)    # h_prev = post_projection out
    tok = sample(logits)                                       # argmax for bit-exact gates
    emit tok
```
Key invariants: **position_ids held constant** across all K steps (L213, L233);
**shared_kv is a fixed snapshot** — the drafter never appends its own drafted
tokens' K/V (it has none). Step 0 seeds `(tok=last_bonus, h_prev=target hidden)`;
step k>0 uses `(tok=prev draft tok, h_prev=drafter post_projection out)`.

### (5) TP-4 implication — the crux (drafter on rank0, all-gather 2 borrowed KV layers)
Under Megatron-style TP-4 of the target:
- **The residual/hidden stream is REPLICATED on every rank** (only in-attention
  per-head + in-MLP per-column are sharded, stitched by all-reduce). ⇒ the
  target's final hidden [5376] (step-0 `recurrent_hidden` seed) is **already on
  every rank — no all-gather needed for the hidden**. Likewise `target.embed`
  and drafter weights (0.5B) can be replicated cheaply.
- **The borrowed K/V is the ONLY thing that is sharded and must be assembled.**
  Target KV is split by kv-heads across ranks: sliding layer 58 → 16 kv-heads =
  4/rank; global layer 59 → 4 kv-heads = 1/rank. The drafter's cross-attention
  needs ALL heads of both borrowed layers.

**Recommended topology (minimizes collectives — the deciding factor on our
comm-expensive vCCL box):**
1. **Drafter runs on rank0 only.** All-gather (concat over the kv-head axis) the
   two borrowed layers' K/V to rank0: sliding {16×kv_len×256}, global
   {4×kv_len×512}. Payload ≈ 24 KB/token (f16, K+V, both layers) — at ctx 2048 ≈
   48 MB, gathered **once per accept cycle** (only the newly-committed-tokens
   delta each cycle, since the drafter re-reads the whole borrowed KV). This is
   **2 layers, not 60** — cheap relative to the target.
2. Rank0 drafts K tokens locally (0.5B, 4 layers — trivial compute; ranks 1-3
   idle, nothing to overlap anyway).
3. **Broadcast the K draft-token ids** (a K-int vector) to all ranks.
4. All 4 ranks run the **TP-4 batched verify** of the K+1 tokens (this is where
   the real target compute + its ~128 all-reduces/tok live, amortized over K —
   same amortization that made the qwen27b-TP4 dense arm a GO).

**Why NOT shard the drafter's attention (option-3):** sharding drafter q-heads
8/rank aligns with the local KV shard (no KV all-gather), BUT it adds an
all-reduce per drafter layer (o_proj is row-parallel) × 4 layers × K steps =
16-32 extra collectives per draft phase — the exact comm pattern our box is worst
at. Option-1's single delta-all-gather + one K-int broadcast per cycle wins
decisively. **This is the load-bearing TP decision; revisit only if the
per-cycle KV all-gather latency is measured to dominate.**

Net: the arm is **tractable** on TP-4 — the drafter is a rank0-local
mini-forward over a gathered 2-layer KV snapshot; no target refactor needed
beyond exposing (a) the replicated final hidden and (b) an all-gather of the two
borrowed KV layers.

---

## 2. `Gemma4Variant::G31b` config — DONE (skeleton + CPU test committed)
Added to `src/model.rs`:
- `Gemma4Variant::G31b` enum variant.
- `Gemma4Config::g31b()` constructor with values read from the REAL
  `text_config` of `gemma-4-31B-it-NVFP4/config.json`.
- CPU test `gemma31b_config_matches_checkpoint` (passes; no GPU/net/checkpoint).

Pinned values (verified against config.json `text_config`):
| field | value | note |
|-------|-------|------|
| hidden_size | 5376 | |
| num_hidden_layers | 60 | |
| num_attention_heads | 32 | |
| num_key_value_heads | 16 | sliding |
| **num_global_key_value_heads** | **4** | **global — differs from G12b's MQA(1)** |
| head_dim / global_head_dim | 256 / 512 | |
| intermediate_size | 21504 | dense, no double-wide |
| num_kv_shared_layers | 0 | no KV-share |
| attention_period | 6 | globals at 5,11,…,59 (10 layers) |
| attention_k_eq_v | true | value-less global (V=K) |
| sliding_window | 1024 | |
| hidden_size_per_layer_input | 0 | no PLE |
| final_logit_softcapping | 30.0 | TARGET only (drafter=null) |
| vocab_size | 262144 | tied embeddings |

**Only real forward-code delta vs G12b:** global layers use 4 KV heads (12B used
1). The forward already reads this via `layer_num_kv_heads`, so **the G31b base
forward is config-only** — no new attention code. `is_full_attention`,
`layer_uses_k_eq_v`, `layer_head_dim`, `layer_intermediate_size` all generalize.

---

## 3. Ordered increment list for the fast-worker (each with its Mac-CPU gate)

> Env for Mac CPU test runs (build then run the test BINARY directly — cargo
> itself dies on the miniforge libiconv shadow):
> ```
> cargo test --lib <name> --no-run
> BIN=$(ls -t target/debug/deps/_rs-* | grep -v '\.d$' | head -1)
> DYLD_LIBRARY_PATH="$HOME/.local/lib:$HOME/miniforge3/lib:/opt/homebrew/lib" "$BIN" <name>
> ```

### INC-0 (DONE) — G31b config + CPU test
Gate: `gemma31b_config_matches_checkpoint` passes. ✔ committed.

### INC-1 — NVFP4 wiring into the gemma loader for the G31b base
Wire the existing NVFP4 GPU path (`GpuWeight::Nvfp4{scales,group_size}`,
`nvfp4_tp_shard`, `nvfp4_fold_scales`, `flags.nvfp4_gpu`, and
`gemma_forward.rs::gemma_res_mv_kind` which ALREADY dispatches
`QuantAux::Nvfp4 => MvKind::Nvfp4` at L32-33) into the gemma resident loader in
`lib.rs` (currently mlx4+q8_0 only, ~L1188-1410). Per the 31B `quantization_config`:
- `language_model.*.mlp.(gate|up|down)_proj` → **NVFP4** (group_16, fp8_e4m3 scales,
  `nvfp4-pack-quantized`).
- `self_attn.(q|k|v|o)_proj` and layers **1,57,58,59** `mlp.(gate|up|down)` →
  **FP8** (per-channel, `float-quantized`, 8-bit) — the loader must branch these
  to the existing `matvec_fp8_variant`/`matvec_fp8_pc` path (gemma_forward.rs L11),
  NOT NVFP4. (config `group_0` targets regex.)
- `lm_head` and `model.embed_tokens` are in `ignore` → tied embed stays host-f16.
- KV cache scheme is fp8 static per-tensor (`kv_cache_scheme`) — note for the
  cluster KV path; not needed for the CPU dequant gate.
Gate (Mac CPU, layer-limited, no GPU): a **dequant bit-exact** unit test — load
one NVFP4 mlp tensor + one FP8 attn tensor from the real 24.7GB shard, CPU-dequant
via the existing reference dequant, and assert cos≥0.9999 vs a stored golden
(reuse the qwen NVFP4 CPU dequant harness). Full-model NVFP4 GPU load is
cluster-deferred.

### INC-1b (DONE) — gemma head-aware TP-4 shard helper + CPU reassembly gate
`load_gemma_nvfp4_weights` refused `VLLM_VULKAN_TP_SIZE>1` (INC-1's note: no
head-aware shard existed — the existing `nvfp4_tp_shard`/`mlx4_tp_shard` in
`src/tp.rs` are typed to `Qwen35Config` and encode qwen's uniform head layout,
not gemma's alternating sliding/global geometry). The 31B base is 24.7GB and
needs TP-4 to fit the cluster's 13.3GB GTT/node.

Landed in `src/tp.rs`:
- `gemma_fp8_tp_shard`/`fp8_shard_rows`/`fp8_shard_cols` — head-aware TP shard
  of the ALWAYS-FP8 `self_attn.{q,k,v,o}_proj` (every layer) + the 4
  FP8-exception `mlp.{gate,up,down}_proj` layers (1/57/58/59). Layer-index-
  aware via `Gemma4Config::layer_head_dim`/`layer_num_kv_heads` — sliding
  layers (16 kv-heads @ 256) and global layers (4 kv-heads @ 512, period-6)
  get DIFFERENT per-rank splits at the same tensor-name pattern.
- `gemma_nvfp4_tp_shard` — head-aware TP shard of the NVFP4
  `mlp.{gate,up,down}_proj` weights (reuses the existing byte/group-aligned
  `nvfp4_shard_rows`/`nvfp4_shard_cols` primitives; every 21504/4=5376 split
  lands on a whole 336-group boundary, a multiple of group_size 16).
- `gemma_tp_shard_f32` — the same head-aware dispatch for the fallback path
  (NVFP4/FP8 declined the packed form and the loader already dequantized to
  f32; only reachable with `VLLM_VULKAN_NVFP4_GPU=0`).
- Wired into `lib.rs`'s `gemma4` (=31B) load block's `on_proj` sink (all 3
  `ProjWeight` arms), and the `VLLM_VULKAN_TP_SIZE>1` refusal replaced with a
  per-head-count divisibility check + rank/size logging (mirrors the qwen3_5
  TP block).

Gate `gemma31b_tp4_shard_reassembly_bitexact` (`src/tp.rs`, ignored by
default, real 24.7GB checkpoint via `VLLM_TEST_GEMMA31B_DIR`): shards each of
5 representative tensors into NR=4 rank-shards, dequantizes each shard
independently, reassembles along the split dim, and asserts **bit-exact
(maxdiff 0.0)** vs dequantizing the full unsharded tensor:
- `self_attn.q_proj` layer 0 (SLIDING, head_dim 256) — maxdiff 0e0.
- `self_attn.q_proj` layer 5 (GLOBAL, head_dim 512, 4 kv-heads) — maxdiff 0e0.
- `self_attn.o_proj` layer 0 (FP8, row-parallel) — maxdiff 0e0.
- `mlp.gate_proj` layer 0 (NVFP4, column-parallel) — maxdiff 0e0.
- `mlp.down_proj` layer 0 (NVFP4, row-parallel) — maxdiff 0e0.

(The NVFP4 reconstruction deliberately uses the ALREADY-FOLDED
`e4m3(weight_scale)*weight_scale_2` single-multiply form — the same form
`gemma_nvfp4_tp_shard`/the real GPU path consume — for both the golden and
every shard, rather than `model::dequantize_nvfp4`'s raw-bytes-plus-separate-
global two-multiply form: `nvfp4_fold_scales_reconstructs_dequant`
(push_constants.rs) already shows those two multiply orderings differ by
~1 ULP under float reassociation, which would make a true maxdiff-0 gate
against `dequantize_nvfp4` impossible and is irrelevant to what this gate
checks — the shard's own partition correctness.)

`cargo test --lib gemma`: 11 passed, 3 ignored (checkpoint-gated), 0 failed —
no regressions.

**GPU/multi-rank EXECUTION of the sharded weights is cluster-deferred** — this
gate proves the partition math CPU-side only (no Vulkan, no multi-rank run).
The cluster milestone is: load `gemma4` with `VLLM_VULKAN_TP_SIZE=4` +
`VLLM_VULKAN_NVFP4_GPU=1` across 4 ranks and confirm coherent/correct output
(the forward's TP dispatch for gemma — analogous to `forward_tp_qwen35` — is
still a follow-up increment; this one only lands the load-time shard).

### INC-2 (DONE) — Drafter loader (bf16 → host-f16, gemma4_assistant tensor map)
New module `src/gemma_assistant.rs`. The drafter checkpoint is **bf16
`model.safetensors` (939MB), NOT quantized** — so port the TENSOR MAP from
`assistant.rs` (L536-700) but load bf16→host-f16 (mirror the g12b host-f16 path,
not the MLX quant loader). Tensor map (48 tensors):
- `model.embed_tokens[262144,1024]` (tied → lm_head), `model.norm[1024]`.
- `pre_projection[1024,10752]`, `post_projection[5376,1024]`.
- layers 0-3: `input_layernorm`, `post_attention_layernorm`,
  `pre_feedforward_layernorm`, `post_feedforward_layernorm` (all [1024]);
  `layer_scalar[1]`; `mlp.{gate[8192,1024],up[8192,1024],down[1024,8192]}`;
  `self_attn.{q_proj, o_proj, q_norm}` — **NO k/v_proj, NO k_norm**.
  layer 3 global: `q_proj[16384,1024]` (32×512), `o_proj[1024,16384]`,
  `q_norm[512]`; layers 0-2 sliding: `q_proj[8192,1024]`, `q_norm[256]`.
Config struct: reuse `AssistantTextConfig` shape from assistant.rs; centroid
path DISABLED (`use_ordered_embeddings:false` → plain tied lm_head).
Gate: a load-shape test — every expected tensor present with the exact shape
above; assert no k_proj/v_proj/k_norm keys exist. Mac-CPU, fits (939MB).

**Landed as `src/gemma_assistant.rs`** (not host-f16: at 939MB the plain f32
widen via the existing `model::load_weights_from_safetensors` bf16 loader
needs no memory-saving trick, and the tensor names already land verbatim in
that loader's namespace — no remap needed). Gate:
`gemma31b_assistant_load_shapes` — loads the real checkpoint, asserts all 48
tensors present with exact shapes, asserts none of k_proj/v_proj/k_norm
exist, asserts no unexpected extra tensors. PASSES.

### INC-3 (DONE) — Drafter CPU forward + bit-exact/cos gate vs reference
Implement the CPU forward mirroring `AssistantModel::forward` (assistant.rs
L382-415) and `draft_block` (gemma4_assistant.py L195-252):
- pre_projection → 4 decoder layers (sandwich norms + layer_scalar at end,
  assistant.rs L265-291) with **Q-only cross-attn over supplied shared K/V**,
  scale=1.0, borrowed-K NOT re-rotated, own-Q rotated+q_norm'd, sliding-window
  mask on layers 0-2 → norm → {post_projection[5376], tied-lm_head logits}.
- `build_inputs_embeds`: concat **embed-first** [prev_token_embed ‖ recurrent].
Gate: feed a TINY fixed input — a synthetic 2-position shared K/V + a seed hidden
+ a bonus token — and compare drafter logits & post_projection output to a golden
dumped from the HF `Gemma4AssistantForCausalLM` (or the MLX drafter) on the same
input. Assert logits cos≥0.999 + argmax match, post_projection cos≥0.999. Golden
generated once via a small python harness against the bf16 checkpoint on Mac.

**Landed as `assistant_forward()` in `src/gemma_assistant.rs`.** Golden
generated by `scripts/gen_gemma31b_assistant_golden.py`, which drives the
REAL upstream `mlx_vlm.models.gemma4.language.{Attention,DecoderLayer}`
classes (`kv_shared_only=True`) directly — wired exactly like
`Gemma4AssistantDraftModel.__call__`/`draft_block` — over the real
checkpoint weights with a synthetic tiny fixed input (kv_len=4, deterministic
PRNG for prev_token_embed/recurrent_hidden/borrowed-K/V), and dumps
`logits.npy`/`post_proj.npy` + the 6 inputs as flat f32 `.npy` fixtures,
committed at `tests/fixtures/gemma31b_assistant_golden/` (1.3MB). Run via the
mlx_vlm venv (`~/.cache/uv/archive-v0/CVdUWPFDsKG7uUgpOGu4v/bin/python3`,
python3.12 — system python3.14 lacks `mlx_vlm`).

Gate `gemma31b_assistant_forward_cos_vs_golden`: **logits cos=1.000000,
argmax rust=42459 == golden=42459, post_projection cos=1.000000** — exact
match (bf16 checkpoint weights widened to f32 on both sides, and the
reference glue code has no other floating-point-order divergence at kv_len=4,
so this is effectively bit-exact rather than merely cos≥0.999). This locks
all 3 of risk #5's silent-failure knobs simultaneously: embed-first concat
order, scale=1.0 (not 1/sqrt(head_dim)), and the KV-borrow (no re-rotation of
borrowed K, own-Q-only RoPE).

### INC-4 — Gemma batched-verify core (12B proxy), CPU bit-exact vs serial
Add `forward_gemma_verify_core(hidden|tokens, start_pos, T)` modeled on
`forward_qwen35_verify_core` (qwen35_forward.rs L3051) but SIMPLER — Gemma is pure
attention, **no GatedDeltaNet, so no recurrent-state capture and rollback = KV
counter rewind only**:
- Run T positions through the decoder stack in one batched pass, appending K/V for
  all T at `start_pos..start_pos+T`; final norm + lm_head + **softcap 30** →
  `T × vocab` logits.
- `gemma_verify_rollback(accept_len)`: commit_len = accept_len+1; if <T, rewind
  each layer's KV seq_len to `start_pos+commit_len` (bytes already correct — no
  re-scan; contrast qwen's GDN re-scan at L3229-3255). This is the big
  simplification vs qwen — call it out.
Gate (Mac CPU, **on the 12B** — 31B does NOT fit Mac's 103GB): assert
`verify_core([t0..tT], R)` row-i logits == the serial
`forward(t_i, R+i)` logits **bit-exact** (or cos=1.0) for a 4-8 token span, and
that after `rollback(a)` the KV frontier == R+a+1 and a subsequent
`forward(tok, R+a+1)` matches the non-spec baseline bit-exact. 12B is the proxy;
identical code runs 31B on cluster.

### INC-5 — EAGLE spec-decode driver (draft K → design-A batched verify → accept/rollback)
`src/gemma_spec.rs` — adapt the qwen design-A driver pattern (spec_pipe.rs
`draft_chain`/accept-reject shapes, L329/L570):
```
loop:
  # step 0 seed: target last hidden (replicated) + bonus token
  draft[0..K] = drafter.draft_block(bonus, target_hidden, shared_kv_snapshot, pos)
  logits[0..K] = gemma_verify_core([bonus, draft[0..K-1]], R, K)   # target, batched
  accept_len = longest prefix where argmax(logits[i]) == draft[i]
  new_bonus  = argmax(logits[accept_len])
  gemma_verify_rollback(accept_len)                                 # KV rewind
  emit committed tokens; R += accept_len+1
```
Gate (Mac CPU, 12B proxy with a STUB drafter or the real 0.5B drafter driving a
12B target-KV-borrow — note the 12B has its OWN last-layer KV of each type, so the
0.5B drafter's borrowed-dim must match; if 12B dims differ, use a stub drafter
that emits fixed ids to exercise the accept/rollback bookkeeping):
**identity gate** — with drafts forced == target argmax, `ids(SPEC)==ids(SPEC=0)`
over ≥32 tokens, and KV frontier tracks exactly. Speedup is cluster-only.

---

## 4. Mac-CPU-validatable NOW vs cluster-deferred

**Mac-CPU NOW (this worktree, CPU-only, no Vulkan):**
- INC-0 G31b config + test ✔.
- INC-1 NVFP4/FP8 loader **dequant bit-exact**, layer-limited (one tensor each).
- INC-2 drafter load-shape test (939MB fits).
- INC-3 drafter CPU forward cos/argmax gate vs HF/MLX golden.
- INC-4 **12B-proxy** batched-verify == serial, bit-exact + rollback frontier.
- INC-5 accept/rollback **bookkeeping** identity gate (stub or 12B-borrow drafter).

**Cluster-deferred (BC-250 TP-4, GFX1013, needs the .so + real 31B):**
- Real 31B TP-4 NVFP4+FP8 **GPU** load (24.7GB → ~6.2GB/rank) & GPU-resident
  matvec correctness (nvfp4 shader GPU-UNVALIDATED — see risks).
- Drafter rank0 residency + the **2-layer borrowed-KV all-gather** + K-int
  broadcast plumbing (INC-5 §1.5 topology).
- The 3-gate A/B: (a) verify==serial bit-exact on 31B; (b) identity
  `ids(SPEC=1)==ids(SPEC=0)`; (c) ≥ speedup threshold (α-dependent).

---

## 5. Risk register
1. **NVFP4-native on GFX1013 is GPU-UNVALIDATED** (base bring-up risk, independent
   of spec-decode). The nvfp4 matvec shader is written+registered but never
   GPU-run (per memory `nvfp4-weight-loading`). If it's wrong, the 31B base is
   wrong before spec-decode enters. Gate INC-1 catches only CPU-dequant math, not
   the GPU shader. **Mitigation:** first cluster milestone = 31B base greedy
   coherence with SPEC off, BEFORE any drafter work lands on cluster.
2. **31B has a MIXED quant layout** (NVFP4 mlp + FP8 attn + FP8 on layers
   1/57/58/59 mlp + fp8 KV cache). The loader must route per-tensor by the
   `quantization_config` regex, not a single format. Mis-routing = silent garbage.
   Borrowed KV for the drafter comes from fp8 KV layers 58/59 → dequant path must
   match what the drafter's scale=1.0 cross-attn expects.
3. **TP-4 KV/hidden plumbing for the drafter** (the §1.5 crux). Two new
   collectives (2-layer KV all-gather + K-int broadcast). The all-gather delta
   must be correct across accept cycles (drafter re-reads whole borrowed KV). If
   the per-cycle gather latency dominates, the arm may not clear the speedup gate
   even with healthy α.
4. **α unknown on our box.** The MLX 27B pair measured 0.43 acceptance / block-8.
   31B α on GFX1013 with our sampling is unmeasured; the speedup gate is
   α-and-comm-dependent and can only be settled on cluster. The qwen27b-dense arm
   was a GO because comm-bound verify amortizes; 31B-dense TP-4 is comm-bound too,
   so the prior is favorable but NOT proven.
5. **Concat order + attention scale are load-bearing and non-obvious.** embed-first
   + scale=1.0 (not 1/sqrt(hd)) are required (assistant.rs L598-608, L706-739);
   getting either wrong silently tanks α to ~0.01 without erroring. INC-3's golden
   gate must lock both.
6. **Drafter global-layer dims (512 hd / 4 kv) differ from sliding (256/16).**
   INC-2/INC-3 must handle the per-layer-type dim split for both q_proj sizing
   and the borrowed-KV shape, exactly as assistant.rs L571-590.
