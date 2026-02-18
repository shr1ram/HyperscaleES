import jax
import optax
import jax.numpy as jnp
from .base_noiser import Noiser

from functools import partial


def _init_lora_leaf(param):
    """Initialize a LoRA leaf: zero matrix for 2D (MM_PARAM), scalar placeholder otherwise."""
    if param.ndim == 2:
        return jnp.zeros_like(param)
    return jnp.zeros((), dtype=param.dtype)


def get_svd_perturbation(frozen_noiser_params, base_sigma, iterinfo, lora, key):
    """Compute SVD perturbation on the LoRA adapter matrix."""
    epoch, thread_id = iterinfo
    r = min(frozen_noiser_params["rank"], min(lora.shape))

    true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"]

    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    # Truncated SVD of the LoRA adapter
    U, S, Vt = jnp.linalg.svd(lora, full_matrices=False)
    U_r = U[:, :r]    # (a, r)
    Vt_r = Vt[:r, :]  # (r, b)

    # Random perturbation in singular value space: only r scalars
    epsilon = jax.random.normal(
        jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx),
        (r,), dtype=lora.dtype
    )

    # delta = epsilon * sigma  (the perturbation to singular values)
    return U_r, Vt_r, epsilon * sigma


def get_nonlora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    epoch, thread_id = iterinfo

    true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"]

    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    updates = jax.random.normal(
        jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx),
        param.shape, dtype=param.dtype
    )
    return updates * sigma


def _simple_full_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    if frozen_noiser_params["freeze_nonlora"]:
        return jnp.zeros_like(param)
    updates = jax.vmap(
        partial(get_nonlora_update_params, frozen_noiser_params),
        in_axes=(None, 0, None, None)
    )(base_sigma, iterinfo, param, key)
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,) * len(param.shape))
    return jnp.astype(jnp.mean(broadcasted_scores * updates, axis=0), param.dtype)


def _noop_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    return jnp.zeros_like(param)


def _lora_svd_update(base_sigma, lora, key, scores, iterinfo, frozen_noiser_params):
    """Update a LoRA adapter using SVD gradient in singular value space."""
    r = min(frozen_noiser_params["rank"], min(lora.shape))

    # Compute SVD once for this LoRA adapter
    U, S, Vt = jnp.linalg.svd(lora, full_matrices=False)
    U_r = U[:, :r]    # (a, r)
    S_r = S[:r]        # (r,)
    Vt_r = Vt[:r, :]  # (r, b)

    # Regenerate all epsilons for each population member: (N, r)
    def _get_epsilon(iterinfo_single):
        epoch, thread_id = iterinfo_single
        true_epoch = jnp.where(frozen_noiser_params["noise_reuse"] == 0, 0, epoch // frozen_noiser_params["noise_reuse"])
        true_thread_idx = thread_id // 2
        sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)
        epsilon = jax.random.normal(
            jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx),
            (r,), dtype=lora.dtype
        )
        return epsilon * sigma

    epochs, thread_ids = iterinfo
    all_epsilons = jax.vmap(_get_epsilon)((epochs, thread_ids))  # (N, r)

    # Gradient in singular value space: weighted average of epsilons
    broadcasted_scores = jnp.reshape(scores, (scores.shape[0], 1))  # (N, 1)
    grad_sigma = jnp.mean(broadcasted_scores * all_epsilons, axis=0)  # (r,)

    # Update singular values: S_new = S - lr * grad (lr is handled by caller scaling)
    S_new = S.at[:r].set(S_r - grad_sigma * jnp.sqrt(scores.size))

    # Reconstruct full LoRA: U @ diag(S_new) @ Vt
    return (U * S_new[None, :]) @ Vt


class Essa(Noiser):
    @classmethod
    def init_noiser(cls, params, sigma, lr, *args, solver=None, solver_kwargs=None, group_size=0, freeze_nonlora=False, noise_reuse=0, rank=1, **kwargs):
        """
        Return frozen_noiser_params and noiser_params.
        Creates a LoRA pytree mirroring params where each 2D leaf gets a zero matrix.
        """
        if solver is None:
            solver = optax.sgd
        if solver_kwargs is None:
            solver_kwargs = {}
        true_solver = solver(lr, **solver_kwargs)
        opt_state = true_solver.init(params)

        # Build LoRA pytree matching params structure
        lora = jax.tree.map(_init_lora_leaf, params)

        return (
            {"group_size": group_size, "freeze_nonlora": freeze_nonlora, "noise_reuse": noise_reuse, "solver": true_solver, "rank": rank},
            {"sigma": sigma, "opt_state": opt_state, "lora": lora}
        )

    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        lora = noiser_params["lora"]  # leaf LoRA matrix for this param (navigated by call_submodule)
        base_ans = x @ param.T
        if iterinfo is None:
            # Eval: apply LoRA with no noise
            return base_ans + x @ lora.T
        # Training: SVD the LoRA adapter, perturb singular values
        U_r, Vt_r, delta = get_svd_perturbation(frozen_noiser_params, noiser_params["sigma"], iterinfo, lora, base_key)
        # Effective forward: base + lora + perturbation
        # lora contribution: x @ lora.T
        # perturbation: x @ Vt_r.T @ diag(delta) @ U_r.T
        return base_ans + x @ lora.T + ((x @ Vt_r.T) * delta) @ U_r.T

    @classmethod
    def do_Tmm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        lora = noiser_params["lora"]  # leaf LoRA matrix for this param
        base_ans = x @ param
        if iterinfo is None:
            # Eval: apply LoRA with no noise
            return base_ans + x @ lora
        # Training: SVD the LoRA adapter, perturb singular values
        U_r, Vt_r, delta = get_svd_perturbation(frozen_noiser_params, noiser_params["sigma"], iterinfo, lora, base_key)
        # Transpose version: base + x @ lora + x @ U_r @ diag(delta) @ Vt_r
        return base_ans + x @ lora + ((x @ U_r) * delta) @ Vt_r

    @classmethod
    def do_emb(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        raise NotImplementedError("Embedding is not implemented")

    @classmethod
    def get_noisy_standard(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo):
        if iterinfo is None or frozen_noiser_params["freeze_nonlora"]:
            return param
        return param + get_nonlora_update_params(frozen_noiser_params, noiser_params["sigma"], iterinfo, param, base_key)

    @classmethod
    def convert_fitnesses(cls, frozen_noiser_params, noiser_params, raw_scores, num_episodes_list=None):
        group_size = frozen_noiser_params["group_size"]
        if group_size == 0:
            true_scores = (raw_scores - jnp.mean(raw_scores, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)
        else:
            group_scores = raw_scores.reshape((-1, group_size))
            true_scores = (group_scores - jnp.mean(group_scores, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)
            true_scores = true_scores.ravel()
        return true_scores

    @classmethod
    def _do_base_update(cls, param, base_key, fitnesses, iterinfos, map_classification, sigma, frozen_noiser_params, **kwargs):
        """Compute gradient for base params (non-LoRA params use full update, MM_PARAM uses noop since LoRA handles them)."""
        update_fn = [_simple_full_update, _noop_update, _noop_update, _noop_update][map_classification]

        if len(base_key.shape) == 0:
            new_grad = update_fn(sigma, param, base_key, fitnesses, iterinfos, frozen_noiser_params)
        else:
            new_grad = jax.lax.scan(lambda _, x: (0, update_fn(sigma, x[0], x[1], fitnesses, iterinfos, frozen_noiser_params)), 0, xs=(param, base_key))[1]

        return -(new_grad * jnp.sqrt(fitnesses.size)).astype(param.dtype)

    @classmethod
    def _do_lora_update(cls, lora, base_key, fitnesses, iterinfos, map_classification, sigma, frozen_noiser_params):
        """Update LoRA adapter via SVD gradient. Only applies to MM_PARAM (map_classification == 1)."""
        if map_classification != 1:
            return lora  # non-MM params: return unchanged placeholder

        if len(base_key.shape) == 0:
            return _lora_svd_update(sigma, lora, base_key, fitnesses, iterinfos, frozen_noiser_params)
        else:
            return jax.lax.scan(
                lambda _, x: (0, _lora_svd_update(sigma, x[0], x[1], fitnesses, iterinfos, frozen_noiser_params)),
                0, xs=(lora, base_key)
            )[1]

    @classmethod
    def do_updates(cls, frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map):
        # 1. Update base params via optax (non-MM params only, MM_PARAM gets noop gradient)
        new_grad = jax.tree.map(
            lambda p, k, m: cls._do_base_update(p, k, fitnesses, iterinfos, m, noiser_params["sigma"], frozen_noiser_params),
            params, base_keys, es_map
        )
        updates, noiser_params["opt_state"] = frozen_noiser_params["solver"].update(new_grad, noiser_params["opt_state"], params)
        new_params = optax.apply_updates(params, updates)

        # 2. Update LoRA adapters directly (SVD-based, not through optax)
        new_lora = jax.tree.map(
            lambda l, k, m: cls._do_lora_update(l, k, fitnesses, iterinfos, m, noiser_params["sigma"], frozen_noiser_params),
            noiser_params["lora"], base_keys, es_map
        )
        noiser_params["lora"] = new_lora

        return noiser_params, new_params
