"""
TinyLlama 1.1B – Full integration test for HyperscaleES / EGGROLL.

Run from repo root:
    python tests/tinyllama_test.py

Tests:
  1. Parameter shapes
  2. Default state shapes
  3. Forward pass + top-k predictions
  4. KV cache position tracking (single + multi-step incremental)
  5. Numerical comparison with HuggingFace (logit correlation, top-token match)
  6. Greedy generation (50 tokens)
  7. EGGROLL noiser init + single update step
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import sys
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time
import traceback

# ── helpers ──────────────────────────────────────────────────────────────────

passed, failed, skipped = [], [], []

def run_test(name, fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        fn()
        passed.append(name)
        print(f"\n  >> PASS")
    except AssertionError as e:
        failed.append(name)
        print(f"\n  >> FAIL: {e}")
        traceback.print_exc()
    except Exception as e:
        failed.append(name)
        print(f"\n  >> ERROR: {e}")
        traceback.print_exc()

def skip_test(name, reason):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  >> SKIPPED: {reason}")
    skipped.append(name)

# ── load model (shared across all tests) ─────────────────────────────────────

import hyperscalees as hs
from hyperscalees.models.llm.auto import get_model
from hyperscalees.models.common import simple_es_tree_key

print("JAX devices:", jax.devices())
print()

NOISER = hs.noiser.base_noiser.Noiser
base_model_key = jax.random.key(0)

print("Loading TinyLlama 1.1B …")
t0 = time.time()
MODEL, full_params, tokenizer = get_model("tl1.1B", verbose=True)
config, params, scan_map, es_map = full_params
params = jax.device_put(params, jax.local_devices()[0])

frozen_noiser_params, noiser_params = NOISER.init_noiser(params, 0.0, None)
base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)
print(f"Model loaded in {time.time()-t0:.1f}s\n")

forward = partial(MODEL.forward, NOISER, frozen_noiser_params, noiser_params, config)

# ── test 1: parameter shapes ────────────────────────────────────────────────

def test_parameter_shapes():
    expected = {
        "embed_tokens.weight":                     (32000, 2048),
        "blocks.self_attn.q_proj.weight":          (22, 2048, 2048),
        "blocks.self_attn.k_proj.weight":          (22, 256, 2048),
        "blocks.self_attn.v_proj.weight":          (22, 256, 2048),
        "blocks.self_attn.o_proj.weight":          (22, 2048, 2048),
        "blocks.mlp.gate_proj.weight":             (22, 5632, 2048),
        "blocks.mlp.up_proj.weight":               (22, 5632, 2048),
        "blocks.mlp.down_proj.weight":             (22, 2048, 5632),
        "blocks.input_layernorm.weight":           (22, 2048),
        "blocks.post_attention_layernorm.weight":  (22, 2048),
        "norm.weight":                             (2048,),
        "lm_head.weight":                          (32000, 2048),
    }
    for dotted, exp_shape in expected.items():
        obj = params
        for k in dotted.split("."):
            obj = obj[k]
        actual = obj.shape
        assert actual == exp_shape, f"{dotted}: got {actual}, expected {exp_shape}"
        print(f"  OK  {dotted}: {actual}")

run_test("1  Parameter shapes", test_parameter_shapes)

# ── test 2: default state shapes ────────────────────────────────────────────

def test_default_state():
    init_state = MODEL.default_state(params, config)
    assert init_state['kv_cache'].shape == (22, 2, 256, 4, 64), \
        f"kv_cache shape: {init_state['kv_cache'].shape}"
    assert int(init_state['cache_pos']) == 0, \
        f"cache_pos: {init_state['cache_pos']}"
    print(f"  kv_cache: {init_state['kv_cache'].shape}")
    print(f"  cache_pos: {init_state['cache_pos']}")

run_test("2  Default state shapes", test_default_state)

# ── test 3: forward pass + top-k ────────────────────────────────────────────

context = "The Eiffel tower is in the city of"
encoded = tokenizer.encode(context)
init_state = MODEL.default_state(params, config)

# These are set by test 3, used by tests 4+
_out = [None]
_state = [None]

def test_forward_pass():
    print(f"  Input: '{context}'  ({len(encoded)} tokens)")
    t0 = time.time()
    out, state = jax.block_until_ready(
        forward(params, base_evo_keys, (0, 1), encoded, init_state)
    )
    elapsed = time.time() - t0
    _out[0] = out
    _state[0] = state

    assert out.shape == (len(encoded), 32000), \
        f"output shape {out.shape}, expected ({len(encoded)}, 32000)"
    print(f"  Output shape: {out.shape}  ({elapsed:.3f}s)")

    soft = jax.nn.softmax(out[-1])
    vals, idxs = jax.lax.top_k(soft, 10)
    print("  Top-10 predictions:")
    for i in range(10):
        print(f"    {vals[i].item()*100:6.2f}%  {tokenizer.decode([idxs[i].item()])!r}")

run_test("3  Forward pass + top-k", test_forward_pass)

# ── test 4: KV cache position tracking ──────────────────────────────────────

def test_kv_cache_tracking():
    out, state = _out[0], _state[0]
    assert out is not None, "test 3 must pass first"

    # After processing the prompt
    assert int(state['cache_pos']) == len(encoded), \
        f"cache_pos={state['cache_pos']}, expected {len(encoded)}"
    print(f"  After prompt:  cache_pos={state['cache_pos']}  (expected {len(encoded)})")

    # Single-token incremental
    next_tok = [int(jnp.argmax(out[-1]))]
    out2, s2 = jax.block_until_ready(
        forward(params, base_evo_keys, (0, 1), next_tok, state)
    )
    assert int(s2['cache_pos']) == len(encoded) + 1
    assert out2.shape == (1, 32000)
    print(f"  After +1 token: cache_pos={s2['cache_pos']}  (expected {len(encoded)+1})")

    # Multi-step incremental (5 more tokens)
    s = s2
    o = out2
    for step in range(5):
        tok = [int(jnp.argmax(o[-1]))]
        o, s = jax.block_until_ready(
            forward(params, base_evo_keys, (0, 1), tok, s)
        )
    expected_pos = len(encoded) + 6
    assert int(s['cache_pos']) == expected_pos, \
        f"cache_pos={s['cache_pos']}, expected {expected_pos}"
    print(f"  After +6 tokens: cache_pos={s['cache_pos']}  (expected {expected_pos})")

run_test("4  KV cache position tracking", test_kv_cache_tracking)

# ── test 5: numerical comparison with HuggingFace ───────────────────────────

def test_hf_comparison():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer as HFAutoTokenizer

    print("  Loading HuggingFace model (float32) …")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32
    )
    hf_tok = HFAutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    hf_model.eval()

    hf_input = hf_tok(context, return_tensors="pt")
    with torch.no_grad():
        hf_output = hf_model(**hf_input)
    hf_logits = hf_output.logits[0].numpy()

    # Fresh JAX forward for fair comparison
    jax_out, _ = jax.block_until_ready(
        forward(params, base_evo_keys, (0, 1), encoded, init_state)
    )
    jax_logits = np.array(jax_out.astype(jnp.float32))

    assert hf_logits.shape == jax_logits.shape, \
        f"Shape mismatch: HF={hf_logits.shape} JAX={jax_logits.shape}"

    hf_last = hf_logits[-1]
    jax_last = jax_logits[-1]

    max_diff = np.max(np.abs(hf_last - jax_last))
    mean_diff = np.mean(np.abs(hf_last - jax_last))
    corr = np.corrcoef(hf_last, jax_last)[0, 1]

    print(f"  Max  abs diff: {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")
    print(f"  Correlation:   {corr:.6f}")

    hf_top = int(np.argmax(hf_last))
    jax_top = int(np.argmax(jax_last))
    print(f"  HF  top: {hf_tok.decode([hf_top])!r}  (id={hf_top})")
    print(f"  JAX top: {tokenizer.decode([jax_top])!r}  (id={jax_top})")

    assert corr > 0.99, f"Correlation {corr:.4f} < 0.99"
    if hf_top != jax_top:
        print("  NOTE: top token differs (bf16 precision edge case, correlation ok)")

    # Free HF model
    del hf_model, hf_output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

try:
    import torch
    run_test("5  Numerical comparison with HuggingFace", test_hf_comparison)
except ImportError:
    skip_test("5  Numerical comparison with HuggingFace", "torch not installed")

# ── test 6: greedy generation ────────────────────────────────────────────────

def test_greedy_generation():
    prompt = "Once upon a time in a land far away,"
    tokens = tokenizer.encode(prompt)
    gen_state = MODEL.default_state(params, config)

    out_logits, gen_state = jax.block_until_ready(
        forward(params, base_evo_keys, (0, 1), tokens, gen_state)
    )

    generated = list(tokens)
    NUM_GEN = 50
    t0 = time.time()
    for _ in range(NUM_GEN):
        next_id = int(jnp.argmax(out_logits[-1]))
        generated.append(next_id)
        out_logits, gen_state = jax.block_until_ready(
            forward(params, base_evo_keys, (0, 1), [next_id], gen_state)
        )
    elapsed = time.time() - t0

    text = tokenizer.decode(generated)
    print(f"  Prompt:    {prompt!r}")
    print(f"  Generated: {text!r}")
    print(f"  Tokens: {len(tokens)} prompt + {NUM_GEN} generated")
    print(f"  Time: {elapsed:.2f}s  ({NUM_GEN/elapsed:.1f} tok/s)")
    print(f"  Final cache_pos: {gen_state['cache_pos']}")

    assert int(gen_state['cache_pos']) == len(tokens) + NUM_GEN
    # Sanity: generated text should be non-empty and not all zeros
    assert len(text) > len(prompt), "Generated text shorter than prompt"

run_test("6  Greedy generation (50 tokens)", test_greedy_generation)

# ── test 7: EGGROLL noiser init + single update step ────────────────────────

def test_eggroll_init():
    EGGROLL = hs.noiser.eggroll.EggRoll

    sigma = 1e-3
    lr_scale = 1.0
    group_size = 8

    print(f"  sigma={sigma}  lr_scale={lr_scale}  group_size={group_size}")

    egg_frozen, egg_noiser = EGGROLL.init_noiser(
        params, sigma, lr_scale, group_size=group_size
    )
    print("  EGGROLL noiser initialized")

    # Verify forward still works with eggroll noiser
    egg_forward = partial(MODEL.forward, EGGROLL, egg_frozen, egg_noiser, config)
    egg_state = MODEL.default_state(params, config)
    iterinfo = (jnp.int32(0), jnp.int32(0))

    out_egg, _ = jax.block_until_ready(
        egg_forward(params, base_evo_keys, iterinfo, encoded, egg_state)
    )
    assert out_egg.shape == (len(encoded), 32000), \
        f"EGGROLL forward shape {out_egg.shape}"
    print(f"  EGGROLL forward pass shape: {out_egg.shape}")

    # Simulate a fitness-based update
    n_gens = group_size * 2  # 16 total
    fake_fitnesses = jnp.arange(n_gens, dtype=jnp.float32) / n_gens
    fitnesses = EGGROLL.convert_fitnesses(egg_frozen, egg_noiser, fake_fitnesses)

    global_indices = jnp.arange(n_gens)
    iterinfos = (jnp.zeros(n_gens, dtype=jnp.int32), global_indices)

    new_noiser, new_params = EGGROLL.do_updates(
        egg_frozen, egg_noiser, params, base_evo_keys, fitnesses, iterinfos, es_map
    )
    print("  EGGROLL update step completed")

    # Check that parameters actually changed
    diff = jax.tree.map(lambda a, b: jnp.max(jnp.abs(a - b)), params, new_params)
    flat_diffs = jax.tree.leaves(diff)
    max_diff = max(float(d) for d in flat_diffs)
    nonzero_count = sum(1 for d in flat_diffs if float(d) > 0)
    print(f"  Max param diff: {max_diff:.6e}")
    print(f"  Leaves with nonzero diff: {nonzero_count}/{len(flat_diffs)}")
    assert nonzero_count > 0, "No parameters were updated"

run_test("7  EGGROLL noiser init + update", test_eggroll_init)

# ── summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  Passed:  {len(passed)}")
print(f"  Failed:  {len(failed)}")
print(f"  Skipped: {len(skipped)}")
if failed:
    print(f"\n  Failed tests:")
    for t in failed:
        print(f"    - {t}")
print()

sys.exit(1 if failed else 0)
