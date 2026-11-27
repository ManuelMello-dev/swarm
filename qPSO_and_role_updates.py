"""Minimal Quantum-inspired Particle Swarm Optimization (QPSO) + QEA rotation-gate
utilities for the ARC proposal swarm.

These are classical simulations of quantum-inspired update rules. They are kept
in a separate module so the main adjudicator stays readable.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import random


class QPSOParticle(NamedTuple):
    """One particle holding a flattened parameter vector."""
    position: jax.Array          # float32 [param_dim]
    personal_best: jax.Array     # float32 [param_dim]
    personal_best_fitness: jax.Array  # scalar


class QPSOState(NamedTuple):
    particles: QPSOParticle      # batched: position [N, D], etc.
    global_best: jax.Array       # [D]
    global_best_fitness: jax.Array
    beta: jax.Array              # contraction-expansion coefficient
    generation: jax.Array


def init_qpso(
    key: jax.Array,
    param_dim: int,
    n_particles: int = 6,
    init_scale: float = 0.05,
) -> QPSOState:
    """Initialize a small QPSO population around the origin (or a seed vector)."""
    keys = random.split(key, n_particles + 1)
    positions = random.normal(keys[0], (n_particles, param_dim), dtype=jnp.float32) * init_scale
    # personal bests start at current positions
    pbest = positions
    pbest_fit = jnp.full((n_particles,), -jnp.inf, dtype=jnp.float32)
    gbest = positions[0]
    gbest_fit = jnp.array(-jnp.inf, dtype=jnp.float32)
    beta = jnp.array(0.7, dtype=jnp.float32)  # typical range 0.5–1.0
    return QPSOState(
        particles=QPSOParticle(position=positions, personal_best=pbest, personal_best_fitness=pbest_fit),
        global_best=gbest,
        global_best_fitness=gbest_fit,
        beta=beta,
        generation=jnp.array(0, dtype=jnp.int32),
    )


def qpso_step(
    state: QPSOState,
    fitness: jax.Array,          # [N] fitness of current positions (higher = better)
    key: jax.Array,
    personal_best_fitness: Optional[jax.Array] = None,  # [N] re-evaluated pbest fitness
    global_best_fitness: Optional[jax.Array] = None,    # scalar re-evaluated gbest fitness
) -> QPSOState:
    """One generation of QPSO (mean-best attractor version).

    Classic QPSO position update (Sun et al.):
        mbest = mean(personal_best)
        p = φ * pbest + (1-φ) * gbest
        x ← p ± β * |mbest - x| * ln(1/u)

    When the objective is non-stationary (each generation scored on a new
    sample), pass the current-sample fitness of the stored personal and global
    bests so comparisons are like-for-like instead of against stale scores.
    """
    N, D = state.particles.position.shape
    pos = state.particles.position
    pbest = state.particles.personal_best
    pbest_fit = (
        state.particles.personal_best_fitness
        if personal_best_fitness is None
        else jnp.asarray(personal_best_fitness, jnp.float32)
    )
    prior_gfit = (
        state.global_best_fitness
        if global_best_fitness is None
        else jnp.asarray(global_best_fitness, jnp.float32)
    )

    # Update personal and global bests
    improved = fitness > pbest_fit
    new_pbest = jnp.where(improved[:, None], pos, pbest)
    new_pbest_fit = jnp.where(improved, fitness, pbest_fit)

    best_idx = jnp.argmax(new_pbest_fit)
    candidate_gbest = new_pbest[best_idx]
    candidate_gfit = new_pbest_fit[best_idx]
    new_gbest = jnp.where(candidate_gfit > prior_gfit, candidate_gbest, state.global_best)
    new_gfit = jnp.maximum(candidate_gfit, prior_gfit)

    # Mean best position
    mbest = jnp.mean(new_pbest, axis=0)

    # Quantum update
    key1, key2, key3 = random.split(key, 3)
    phi = random.uniform(key1, (N, 1), dtype=jnp.float32)          # [0,1]
    u = random.uniform(key2, (N, 1), dtype=jnp.float32) + 1e-8
    sign = jnp.where(random.uniform(key3, (N, 1)) < 0.5, 1.0, -1.0)

    p = phi * new_pbest + (1.0 - phi) * new_gbest[None, :]
    # β can be annealed; here we keep it constant for simplicity
    beta = state.beta
    new_pos = p + sign * beta * jnp.abs(mbest[None, :] - pos) * jnp.log(1.0 / u)

    return QPSOState(
        particles=QPSOParticle(
            position=new_pos,
            personal_best=new_pbest,
            personal_best_fitness=new_pbest_fit,
        ),
        global_best=new_gbest,
        global_best_fitness=new_gfit,
        beta=beta,
        generation=state.generation + 1,
    )


# ---------------------------------------------------------------------------
# Quantum-inspired Evolutionary Algorithm (QEA) rotation-gate update
# ---------------------------------------------------------------------------

def rotation_gate_update(
    amplitudes: jax.Array,          # complex or real [..., 2]  (alpha, beta) or just real bias
    fitness_delta: jax.Array,       # scalar or broadcastable – positive → rotate toward 1
    theta_max: float = 0.05 * jnp.pi,
) -> jax.Array:
    """Classic QEA rotation-gate style update for a real-valued bias vector.

    For continuous parameters we use a soft version:
        bias ← bias + Δθ * sign(fitness_delta) * (target - current_normalized)

    This is a lightweight continuous analogue of the binary rotation gate.
    """
    # Normalize current values to roughly [-1,1] for stable rotation
    scale = jnp.maximum(jnp.std(amplitudes), 1e-3)
    normalized = amplitudes / scale
    # Target is +1 when fitness improved, -1 otherwise
    target = jnp.sign(fitness_delta)
    delta = theta_max * target * (1.0 - jnp.abs(normalized))  # larger step when far from target
    return amplitudes + delta * scale


def role_rotation_update(
    role_bias: jax.Array,           # [num_roles, num_hypotheses]
    role_fitness: jax.Array,        # [num_roles]  (higher = better)
    prev_role_fitness: jax.Array,   # [num_roles]
    theta_max: float = 0.08,
) -> jax.Array:
    """Apply a rotation-gate-style update to each role's hypothesis bias.

    Roles that improved their fitness are gently rotated toward the hypotheses
    they currently prefer; roles that worsened are rotated away.
    """
    delta = role_fitness - prev_role_fitness
    # For each role, compute a soft target direction from its current bias
    # (encourage the currently strongest hypotheses a little more)
    strength = jax.nn.softmax(role_bias, axis=-1)
    # Update magnitude proportional to improvement
    step = theta_max * jnp.tanh(delta)[:, None]   # [R, 1]
    # Move bias toward the current soft-max direction when improved
    update = step * (strength - 1.0 / role_bias.shape[-1])
    return role_bias + update
