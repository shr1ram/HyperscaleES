import jax
import jax.numpy as jnp

from functools import partial

from .llm import LLM
from ..base_model import Model, CommonParams
from ..common import PARAM, MM_PARAM, EMB_PARAM, EXCLUDED, Parameter, MM, TMM, Embedding, Linear, call_submodule


# --- Pure RoPE functions (no learnable params) ---

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q: (T, n_heads, head_dim), k: (T, n_kv_heads, head_dim)
    # cos, sin: (max_seq_len, head_dim/2) from precompute
    # position_ids: (T,) absolute position indices
    cos_pos = cos[position_ids]  # (T, head_dim/2)
    sin_pos = sin[position_ids]  # (T, head_dim/2)
    # Double up to full head_dim by concatenating
    cos_pos = jnp.concatenate([cos_pos, cos_pos], axis=-1)[:, None, :]  # (T, 1, head_dim)
    sin_pos = jnp.concatenate([sin_pos, sin_pos], axis=-1)[:, None, :]  # (T, 1, head_dim)
    q_embed = q * cos_pos + rotate_half(q) * sin_pos
    k_embed = k * cos_pos + rotate_half(k) * sin_pos
    return q_embed, k_embed


# --- Model submodules ---

class LlamaRMSNorm(Model):
    @classmethod
    def _forward(cls, common_params, x, eps=1e-5):
        hidden_states = x
        variance = jnp.mean(hidden_states ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + eps)
        return call_submodule(Parameter, 'weight', common_params) * hidden_states


class LlamaMLP(Model):
    @classmethod
    def _forward(cls, common_params, x):
        return call_submodule(Linear, 'down_proj', common_params,
                              jax.nn.silu(call_submodule(Linear, 'gate_proj', common_params, x)) *
                              call_submodule(Linear, 'up_proj', common_params, x))


class LlamaAttention(Model):
    @classmethod
    def _forward(cls, common_params, x, kv_cache, cache_pos, n_heads, n_kv_heads, head_dim, rope_cos, rope_sin):
        T, _ = x.shape
        num_kv_groups = n_heads // n_kv_heads

        # Project Q, K, V
        q = call_submodule(Linear, 'q_proj', common_params, x).reshape(T, n_heads, head_dim)
        k = call_submodule(Linear, 'k_proj', common_params, x).reshape(T, n_kv_heads, head_dim)
        v = call_submodule(Linear, 'v_proj', common_params, x).reshape(T, n_kv_heads, head_dim)

        # Apply RoPE using absolute positions
        position_ids = cache_pos + jnp.arange(T)
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin, position_ids)

        # Write new K/V into cache at cache_pos positions
        # kv_cache: (2, max_seq_len, n_kv_heads, head_dim)
        indices = cache_pos + jnp.arange(T)
        kv_cache = kv_cache.at[0, indices].set(k)
        kv_cache = kv_cache.at[1, indices].set(v)

        # Read full cache up to current position
        max_seq_len = kv_cache.shape[1]
        valid_len = cache_pos + T

        # Cached keys and values: (max_seq_len, n_kv_heads, head_dim)
        cached_k = kv_cache[0]
        cached_v = kv_cache[1]

        # Repeat KV heads for GQA
        cached_k = jnp.repeat(cached_k, num_kv_groups, axis=1)  # (max_seq_len, n_heads, head_dim)
        cached_v = jnp.repeat(cached_v, num_kv_groups, axis=1)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        # q: (T, n_heads, head_dim), cached_k: (max_seq_len, n_heads, head_dim)
        # attn_weights: (n_heads, T, max_seq_len)
        attn_weights = jnp.einsum('thd,shd->hts', q, cached_k) * scale

        # Causal mask: query at position i can attend to key at position j if j <= i
        q_positions = cache_pos + jnp.arange(T)  # (T,)
        k_positions = jnp.arange(max_seq_len)    # (max_seq_len,)
        causal_mask = q_positions[:, None] >= k_positions[None, :]  # (T, max_seq_len)

        # Validity mask: only attend to positions that have been written
        valid_mask = k_positions[None, :] < valid_len  # (1, max_seq_len)

        # Combine masks
        mask = causal_mask & valid_mask  # (T, max_seq_len)
        attn_weights = jnp.where(mask[None, :, :], attn_weights, jnp.finfo(attn_weights.dtype).min)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(x.dtype)

        # Apply attention: (n_heads, T, max_seq_len) @ (max_seq_len, n_heads, head_dim) -> (T, n_heads, head_dim)
        attn_output = jnp.einsum('hts,shd->thd', attn_weights, cached_v)

        # Reshape and project output
        attn_output = attn_output.reshape(T, n_heads * head_dim)
        output = call_submodule(Linear, 'o_proj', common_params, attn_output)

        return output, kv_cache


# --- Main model classes ---

class BaseTinyLlama(LLM):
    @classmethod
    def transform_torch_model(cls, torch_model, dtype=jnp.bfloat16):
        import torch
        w = torch_model
        keys = list(w.keys())
        for k in keys:
            k_new = k.replace("model.", "").replace("layers.", "blocks.")
            if k_new != k:
                w[k_new] = w[k]
                del w[k]
        # If lm_head is tied to embed_tokens, copy it
        if 'lm_head.weight' not in w:
            w['lm_head.weight'] = w['embed_tokens.weight'].clone()
        return w

    @classmethod
    def transform_config(cls, config):
        if config is None:
            config = {}
        # Precompute RoPE tables and inject into config
        head_dim = config.get('head_dim', 64)
        max_seq_len = config.get('max_seq_len', 256)
        rope_theta = config.get('rope_theta', 10000.0)
        cos, sin = precompute_freqs_cis(head_dim, max_seq_len, rope_theta)
        config['rope_cos'] = cos
        config['rope_sin'] = sin
        return config

    @classmethod
    def get_scan_map(cls, config):
        BS = (0,)
        NS = tuple()
        return {
            'blocks': {
                'input_layernorm': {'weight': BS},
                'mlp': {
                    'down_proj': {'weight': BS},
                    'gate_proj': {'weight': BS},
                    'up_proj': {'weight': BS},
                },
                'post_attention_layernorm': {'weight': BS},
                'self_attn': {
                    'q_proj': {'weight': BS},
                    'k_proj': {'weight': BS},
                    'v_proj': {'weight': BS},
                    'o_proj': {'weight': BS},
                },
            },
            'embed_tokens': {'weight': NS},
            'lm_head': {'weight': NS},
            'norm': {'weight': NS},
        }

    @classmethod
    def get_es_map(cls, config):
        LORA = MM_PARAM
        FULL = PARAM
        return {
            'blocks': {
                'input_layernorm': {'weight': FULL},
                'mlp': {
                    'down_proj': {'weight': LORA},
                    'gate_proj': {'weight': LORA},
                    'up_proj': {'weight': LORA},
                },
                'post_attention_layernorm': {'weight': FULL},
                'self_attn': {
                    'q_proj': {'weight': LORA},
                    'k_proj': {'weight': LORA},
                    'v_proj': {'weight': LORA},
                    'o_proj': {'weight': LORA},
                },
            },
            'embed_tokens': {'weight': EXCLUDED},
            'lm_head': {'weight': EXCLUDED},
            'norm': {'weight': FULL},
        }

    @classmethod
    def default_state(cls, params, config):
        n_layer = config['n_layer']
        n_kv_heads = config['n_kv_heads']
        head_dim = config['head_dim']
        max_seq_len = config['max_seq_len']
        dtype = params['embed_tokens']['weight'].dtype
        return {
            'kv_cache': jnp.zeros((n_layer, 2, max_seq_len, n_kv_heads, head_dim), dtype=dtype),
            'cache_pos': jnp.array(0, dtype=jnp.int32),
        }

    @classmethod
    def embed(cls, common_params, tokens):
        return common_params.params['embed_tokens']['weight'][tokens.ravel()]

    @classmethod
    def outhead(cls, common_params, x):
        x = call_submodule(LlamaRMSNorm, 'norm', common_params, x,
                           eps=common_params.frozen_params.get('rms_norm_eps', 1e-5))
        return x @ common_params.params['lm_head']['weight'].T

    @classmethod
    def forward_seq(cls, common_params, x, state, length, new_starts):
        config = common_params.frozen_params
        n_heads = config['n_heads']
        n_kv_heads = config['n_kv_heads']
        head_dim = config['head_dim']
        rms_norm_eps = config.get('rms_norm_eps', 1e-5)
        rope_cos = config['rope_cos']
        rope_sin = config['rope_sin']

        kv_cache = state['kv_cache']
        cache_pos = state['cache_pos']

        @partial(jax.checkpoint,
                 policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        def block_loop(x, inputs):
            hidden_states = x
            params_i, es_tree_key_i, kv_cache_i = inputs
            block_i = common_params._replace(
                params=params_i,
                es_tree_key=es_tree_key_i
            )

            # Self attention
            residual = hidden_states
            hidden_states = call_submodule(LlamaRMSNorm, 'input_layernorm', block_i, hidden_states, eps=rms_norm_eps)
            hidden_states, kv_cache_i = call_submodule(LlamaAttention, 'self_attn', block_i,
                                                       hidden_states, kv_cache_i, cache_pos,
                                                       n_heads, n_kv_heads, head_dim,
                                                       rope_cos, rope_sin)
            hidden_states = residual + hidden_states

            # MLP
            residual = hidden_states
            hidden_states = call_submodule(LlamaRMSNorm, 'post_attention_layernorm', block_i, hidden_states, eps=rms_norm_eps)
            hidden_states = call_submodule(LlamaMLP, 'mlp', block_i, hidden_states)
            hidden_states = residual + hidden_states

            return hidden_states, kv_cache_i

        x, kv_cache = jax.lax.scan(block_loop, x,
                                    (common_params.params['blocks'],
                                     common_params.es_tree_key['blocks'],
                                     kv_cache))

        T = x.shape[0]
        new_state = {
            'kv_cache': kv_cache,
            'cache_pos': cache_pos + T,
        }
        return x, new_state


class FastTinyLlama(BaseTinyLlama):
    @classmethod
    def forward_seq(cls, common_params, x, state, length, new_starts):
        config = common_params.frozen_params
        n_layer = config['n_layer']
        n_heads = config['n_heads']
        n_kv_heads = config['n_kv_heads']
        head_dim = config['head_dim']
        rms_norm_eps = config.get('rms_norm_eps', 1e-5)
        rope_cos = config['rope_cos']
        rope_sin = config['rope_sin']

        kv_cache = state['kv_cache']
        cache_pos = state['cache_pos']

        for i in range(n_layer):
            params_i = jax.tree.map(lambda a: a[i], common_params.params['blocks'])
            es_tree_key_i = jax.tree.map(lambda a: a[i], common_params.es_tree_key['blocks'])
            kv_cache_i = kv_cache[i]
            block_i = common_params._replace(
                params=params_i,
                es_tree_key=es_tree_key_i
            )

            # Self attention
            residual = x
            hidden_states = call_submodule(LlamaRMSNorm, 'input_layernorm', block_i, x, eps=rms_norm_eps)
            hidden_states, kv_cache_i = call_submodule(LlamaAttention, 'self_attn', block_i,
                                                       hidden_states, kv_cache_i, cache_pos,
                                                       n_heads, n_kv_heads, head_dim,
                                                       rope_cos, rope_sin)
            x = residual + hidden_states

            # MLP
            residual = x
            hidden_states = call_submodule(LlamaRMSNorm, 'post_attention_layernorm', block_i, x, eps=rms_norm_eps)
            hidden_states = call_submodule(LlamaMLP, 'mlp', block_i, hidden_states)
            x = residual + hidden_states

            kv_cache = kv_cache.at[i].set(kv_cache_i)

        T = x.shape[0]
        new_state = {
            'kv_cache': kv_cache,
            'cache_pos': cache_pos + T,
        }
        return x, new_state
