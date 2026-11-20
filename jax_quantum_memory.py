"""Pure-JAX bounded associative memory for the ARC swarm.

The memory is a classical simulation of normalized complex state vectors. All
recall and insertion math is JAX-array computation; host synchronization is
limited to explicit diagnostics such as ``get_stats``.

This revision makes the feature→amplitude projection learnable.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax, random


class JAXMemoryState(NamedTuple):
    amplitudes: jax.Array       # complex64 [capacity, dim]
    emotional_weights: jax.Array  # float32 [capacity]
    phases: jax.Array           # float32 [capacity]
    task_ids: jax.Array         # int32 [capacity]
    action_ids: jax.Array       # int32 [capacity], 1..6
    rewards: jax.Array          # float32 [capacity]
    valid: jax.Array            # bool [capacity]
    cursor: jax.Array           # int32 scalar
    count: jax.Array           # int32 scalar
    total_encoded: jax.Array    # int32 scalar
    total_recalled: jax.Array   # int32 scalar
    features: jax.Array        # float32 [capacity, FEATURE_DIM]
    feature_valid: jax.Array   # bool [capacity]


# Feature dimension produced by the hand-crafted extractor
FEATURE_DIM = 73  # 10 hist + 40 moments + 16 pool4 + 4 pool2 + 3 shape/edge


def create_memory_state(capacity: int, dim: int) -> JAXMemoryState:
    if capacity < 1 or dim < 1:
        raise ValueError("capacity and dim must be positive")
    return JAXMemoryState(
        amplitudes=jnp.zeros((capacity, dim), jnp.complex64),
        emotional_weights=jnp.zeros((capacity,), jnp.float32),
        phases=jnp.zeros((capacity,), jnp.float32),
        task_ids=jnp.full((capacity,), -1, jnp.int32),
        action_ids=jnp.zeros((capacity,), jnp.int32),
        rewards=jnp.zeros((capacity,), jnp.float32),
        valid=jnp.zeros((capacity,), jnp.bool_),
        cursor=jnp.array(0, jnp.int32),
        count=jnp.array(0, jnp.int32),
        total_encoded=jnp.array(0, jnp.int32),
        total_recalled=jnp.array(0, jnp.int32),
        features=jnp.zeros((capacity, FEATURE_DIM), jnp.float32),
        feature_valid=jnp.zeros((capacity,), jnp.bool_),
    )


def normalize_amplitudes(amplitudes: jax.Array) -> jax.Array:
    amplitudes = jnp.asarray(amplitudes, dtype=jnp.complex64)
    squared_norm = jnp.sum(jnp.square(jnp.real(amplitudes)) + jnp.square(jnp.imag(amplitudes)))
    norm = jnp.sqrt(squared_norm + jnp.asarray(1e-30, jnp.float32))
    normalized = amplitudes / norm
    basis = jax.nn.one_hot(0, amplitudes.shape[0], dtype=jnp.float32).astype(jnp.complex64)
    return jnp.where(squared_norm > 1e-30, normalized, basis)


def project_features(
    features: jax.Array, proj_real: jax.Array, proj_imag: jax.Array
) -> jax.Array:
    """Map one FEATURE_DIM vector through the current complex projection."""
    features = jnp.asarray(features, dtype=jnp.float32)
    real = features @ proj_real
    imag = features @ proj_imag
    return normalize_amplitudes((real + 1j * imag).astype(jnp.complex64))


def extract_grid_features(grid: jax.Array) -> jax.Array:
    """Hand-crafted real features (length FEATURE_DIM). Pure device function."""
    grid = jnp.asarray(grid, dtype=jnp.float32)
    grid = jnp.nan_to_num(grid, nan=0.0, posinf=9.0, neginf=0.0)
    grid = jnp.clip(jnp.rint(grid), 0, 9)

    if grid.size == 0:
        raise ValueError("grid must contain at least one value")

    if grid.ndim == 1:
        grid = grid.reshape(1, -1)
    elif grid.ndim > 2:
        grid = grid.reshape(grid.shape[0], -1)

    h, w = grid.shape
    num_colors = 10

    flat = grid.reshape(-1).astype(jnp.int32)
    hist = jnp.bincount(flat, length=num_colors).astype(jnp.float32)
    hist = hist / jnp.maximum(jnp.sum(hist), 1.0)

    yy, xx = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, h, dtype=jnp.float32),
        jnp.linspace(-1.0, 1.0, w, dtype=jnp.float32),
        indexing="ij",
    )
    one_hot = jax.nn.one_hot(grid.astype(jnp.int32), num_colors, dtype=jnp.float32)
    mass = jnp.sum(one_hot, axis=(0, 1)) + 1e-6

    cx = jnp.sum(one_hot * xx[..., None], axis=(0, 1)) / mass
    cy = jnp.sum(one_hot * yy[..., None], axis=(0, 1)) / mass
    mean_color = jnp.sum(one_hot * grid[..., None], axis=(0, 1)) / mass
    max_color = jnp.max(one_hot * grid[..., None], axis=(0, 1))

    moments = jnp.concatenate([cx, cy, mean_color, max_color])  # 40

    def pool(size: int) -> jax.Array:
        pooled = jax.image.resize(grid, (size, size), method="area")
        return pooled.reshape(-1)

    pool4 = pool(4)
    pool2 = pool(2)

    dx = jnp.diff(grid, axis=1, prepend=grid[:, :1])
    dy = jnp.diff(grid, axis=0, prepend=grid[:1, :])
    edge = jnp.mean(jnp.abs(dx) + jnp.abs(dy))

    features = jnp.concatenate([
        hist,
        moments,
        pool4,
        pool2,
        jnp.asarray([edge, h / 30.0, w / 30.0], jnp.float32),
    ])
    return features  # FEATURE_DIM


def state_from_grid(
    grid: jax.Array,
    dim: int,
    proj_real: Optional[jax.Array] = None,
    proj_imag: Optional[jax.Array] = None,
) -> jax.Array:
    """Map grid → normalized complex state using (optionally learnable) projection.

    If proj_real / proj_imag are None, a deterministic fixed projection is used
    (backward compatible). Otherwise they must be float32 [FEATURE_DIM, dim].
    """
    if dim < 1:
        raise ValueError("dim must be positive")

    features = extract_grid_features(grid)

    if proj_real is None or proj_imag is None:
        # Deterministic fallback (reproducible)
        key = random.PRNGKey(42)
        scale = 1.0 / jnp.sqrt(float(FEATURE_DIM))
        proj_real = random.normal(key, (FEATURE_DIM, dim), dtype=jnp.float32) * scale
        key, subkey = random.split(key)
        proj_imag = random.normal(subkey, (FEATURE_DIM, dim), dtype=jnp.float32) * scale

    return project_features(features, proj_real, proj_imag)


def init_projection(dim: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Initialize learnable real/imag projection matrices."""
    scale = 1.0 / jnp.sqrt(float(FEATURE_DIM))
    key1, key2 = random.split(key)
    proj_real = random.normal(key1, (FEATURE_DIM, dim), dtype=jnp.float32) * scale
    proj_imag = random.normal(key2, (FEATURE_DIM, dim), dtype=jnp.float32) * scale
    return proj_real, proj_imag


def jax_memory_encode(
    state: JAXMemoryState,
    amplitudes: jax.Array,
    action_id: int | jax.Array,
    reward: float | jax.Array,
    task_id: int | jax.Array,
    phase: float | jax.Array,
    emotional_weight: float | jax.Array = 1.0,
    features: Optional[jax.Array] = None,
) -> JAXMemoryState:
    """Insert one experience into a fixed-capacity ring, entirely on device."""
    vector = normalize_amplitudes(amplitudes)
    feature_vector = (
        jnp.zeros((FEATURE_DIM,), jnp.float32)
        if features is None
        else jnp.asarray(features, dtype=jnp.float32)
    )
    has_features = jnp.asarray(features is not None, dtype=jnp.bool_)
    slot = state.cursor
    return state._replace(
        amplitudes=state.amplitudes.at[slot].set(vector),
        features=state.features.at[slot].set(feature_vector),
        feature_valid=state.feature_valid.at[slot].set(has_features),
        emotional_weights=state.emotional_weights.at[slot].set(
            jnp.clip(jnp.asarray(emotional_weight, jnp.float32), 0.0, 1.0)
        ),
        phases=state.phases.at[slot].set(jnp.asarray(phase, jnp.float32)),
        task_ids=state.task_ids.at[slot].set(jnp.asarray(task_id, jnp.int32)),
        action_ids=state.action_ids.at[slot].set(jnp.asarray(action_id, jnp.int32)),
        rewards=state.rewards.at[slot].set(jnp.asarray(reward, jnp.float32)),
        valid=state.valid.at[slot].set(True),
        cursor=(slot + 1) % state.amplitudes.shape[0],
        count=jnp.minimum(state.count + 1, state.amplitudes.shape[0]),
        total_encoded=state.total_encoded + 1,
    )


def jax_memory_recall(
    state: JAXMemoryState,
    query: jax.Array,
    phase: float | jax.Array,
    top_k: int,
    use_phi_drift: bool,
    cross_task_weight: float,
    task_id: int | jax.Array,
    decay_rate: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return fixed-width recalled features, phase-weighted scores, and slots."""
    capacity, dim = state.amplitudes.shape
    if top_k < 1 or top_k > capacity:
        raise ValueError("top_k must be between one and memory capacity")
    q = jnp.asarray(query, jnp.complex64)
    qnorm = jnp.sum(jnp.abs(q) ** 2)
    mnorm = jnp.sum(jnp.abs(state.amplitudes) ** 2, axis=-1)
    overlap = jnp.sum(jnp.conj(q)[None, :] * state.amplitudes, axis=-1)
    fidelity = jnp.square(jnp.abs(overlap)) / jnp.maximum(qnorm * mnorm, 1e-30)
    fidelity = jnp.where((qnorm > 1e-30) & (mnorm > 1e-30), fidelity, 0.0)

    phase_delta = jnp.arctan2(
        jnp.sin(jnp.asarray(phase) - state.phases),
        jnp.cos(jnp.asarray(phase) - state.phases),
    )
    phase_factor = 0.5 + 0.5 * (1.0 + jnp.cos(phase_delta)) / 2.0
    age = jnp.maximum(jnp.asarray(phase, jnp.float32) - state.phases, 0.0)
    temporal = jnp.exp(-jnp.asarray(decay_rate, jnp.float32) * age)
    phase_weight = phase_factor if use_phi_drift else 1.0
    task_factor = jnp.where(
        state.task_ids == jnp.asarray(task_id, jnp.int32),
        1.0,
        jnp.asarray(cross_task_weight, jnp.float32),
    )
    scores_all = fidelity * state.emotional_weights * temporal * phase_weight * task_factor
    scores_all = jnp.where(state.valid, scores_all, -jnp.inf)
    scores, indices = lax.top_k(scores_all, top_k)
    selected_valid = state.valid[indices]
    safe_scores = jnp.where(selected_valid, jnp.maximum(scores, 0.0), 0.0)
    weights = safe_scores / jnp.maximum(jnp.sum(safe_scores), 1e-12)
    selected_amplitudes = state.amplitudes[indices]
    interference = jnp.sum(weights[:, None] * selected_amplitudes, axis=0)
    action_ids = state.action_ids[indices]
    action_features = jnp.sum(
        weights[:, None] * jax.nn.one_hot(jnp.maximum(action_ids - 1, 0), 6), axis=0
    )
    reward_feature = jnp.sum(weights * state.rewards[indices])
    best_score = jnp.max(safe_scores)
    features = jnp.concatenate((
        jnp.real(interference).astype(jnp.float32),
        jnp.imag(interference).astype(jnp.float32),
        action_features.astype(jnp.float32),
        jnp.asarray([reward_feature, best_score], jnp.float32),
    ))
    return features, safe_scores, indices


def _projection_train_step(
    proj_real: jax.Array,
    proj_imag: jax.Array,
    state: JAXMemoryState,
    query_features: jax.Array,
    task_id: jax.Array,
    learning_rate: jax.Array,
    cross_task_weight: float,
    similarity_temperature: float,
    reward_temperature: float,
    grad_clip_norm: float,
) -> tuple[jax.Array, jax.Array, JAXMemoryState, jax.Array, jax.Array, jax.Array]:
    """Train the projection against outcome-weighted memories, then refresh them."""
    trainable = state.valid & state.feature_valid

    def update(_):
        def loss_fn(real_matrix, imag_matrix):
            query = project_features(query_features, real_matrix, imag_matrix)
            memories = jax.vmap(
                lambda feature: project_features(feature, real_matrix, imag_matrix)
            )(state.features)
            overlap = jnp.sum(jnp.conj(query)[None, :] * memories, axis=-1)
            fidelity = jnp.square(jnp.abs(overlap))
            similarity_logits = jnp.log(fidelity + 1e-8) / similarity_temperature

            task_factor = jnp.where(
                state.task_ids == jnp.asarray(task_id, jnp.int32),
                1.0,
                jnp.asarray(cross_task_weight, jnp.float32),
            )
            target_logits = (
                jnp.clip(state.rewards, -5.0, 5.0) / reward_temperature
                + jnp.log(state.emotional_weights + 1e-6)
                + jnp.log(task_factor + 1e-6)
            )
            masked_target = jnp.where(trainable, target_logits, -1e9)
            masked_similarity = jnp.where(trainable, similarity_logits, -1e9)
            target = jax.nn.softmax(masked_target)
            log_prediction = jax.nn.log_softmax(masked_similarity)
            return -jnp.sum(target * log_prediction)

        loss, (grad_real, grad_imag) = jax.value_and_grad(
            loss_fn, argnums=(0, 1)
        )(proj_real, proj_imag)
        grad_norm = jnp.sqrt(
            jnp.sum(jnp.square(grad_real)) + jnp.sum(jnp.square(grad_imag))
        )
        scale = jnp.minimum(1.0, grad_clip_norm / (grad_norm + 1e-12))
        next_real = proj_real - learning_rate * scale * grad_real
        next_imag = proj_imag - learning_rate * scale * grad_imag

        refreshed = jax.vmap(
            lambda feature: project_features(feature, next_real, next_imag)
        )(state.features)
        next_amplitudes = jnp.where(trainable[:, None], refreshed, state.amplitudes)
        next_state = state._replace(amplitudes=next_amplitudes)
        return next_real, next_imag, next_state, loss, grad_norm, jnp.asarray(True)

    def skip(_):
        zero = jnp.asarray(0.0, jnp.float32)
        return proj_real, proj_imag, state, zero, zero, jnp.asarray(False)

    can_train = jnp.sum(trainable.astype(jnp.int32)) >= 2
    return lax.cond(can_train, update, skip, operand=None)


class JAXQuantumMemoryField:
    """Mutable host owner for a JAX-resident fixed-capacity memory state.

    Now also owns the learnable feature→amplitude projection.
    """

    def __init__(
        self,
        dim: int = 32,
        capacity: int = 1000,
        top_k: int = 5,
        decay_rate: float = 0.01,
        cross_task_weight: float = 0.25,
        use_phi_drift: bool = True,
        seed: int = 42,
        projection_similarity_temperature: float = 0.1,
        projection_reward_temperature: float = 0.5,
        projection_grad_clip_norm: float = 1.0,
    ):
        if not 1 <= top_k <= capacity:
            raise ValueError("top_k must be between one and capacity")
        if decay_rate < 0 or not 0.0 <= cross_task_weight <= 1.0:
            raise ValueError("invalid decay_rate or cross_task_weight")
        if min(projection_similarity_temperature, projection_reward_temperature,
               projection_grad_clip_norm) <= 0:
            raise ValueError("projection temperatures and grad clip norm must be positive")
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.top_k = int(top_k)
        self.decay_rate = float(decay_rate)
        self.cross_task_weight = float(cross_task_weight)
        self.use_phi_drift = bool(use_phi_drift)
        self.projection_similarity_temperature = float(projection_similarity_temperature)
        self.projection_reward_temperature = float(projection_reward_temperature)
        self.projection_grad_clip_norm = float(projection_grad_clip_norm)
        self.state = create_memory_state(capacity, dim)

        # Learnable projection
        key = random.PRNGKey(seed)
        self.proj_real, self.proj_imag = init_projection(dim, key)

        self._recall = jax.jit(
            lambda st, q, phase, task: jax_memory_recall(
                st, q, phase, self.top_k, self.use_phi_drift,
                self.cross_task_weight, task, self.decay_rate,
            )
        )
        self._encode = jax.jit(jax_memory_encode)
        self._projection_train = jax.jit(
            lambda real, imag, state, query, task, rate: _projection_train_step(
                real, imag, state, query, task, rate,
                self.cross_task_weight,
                self.projection_similarity_temperature,
                self.projection_reward_temperature,
                self.projection_grad_clip_norm,
            )
        )

    def encode_grid(self, grid: jax.Array) -> jax.Array:
        """Convenience: grid → amplitudes using current learnable projection."""
        return self.encode_grid_with_features(grid)[1]

    def encode_grid_with_features(self, grid: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return both the raw feature vector and its current amplitude embedding."""
        features = extract_grid_features(grid)
        return features, project_features(features, self.proj_real, self.proj_imag)

    def update_projection(
        self, query_features: jax.Array, task_id: int, learning_rate: float
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Update both projection matrices and re-embed feature-bearing memories."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        (
            self.proj_real, self.proj_imag, self.state,
            loss, grad_norm, updated,
        ) = self._projection_train(
            self.proj_real, self.proj_imag, self.state,
            jnp.asarray(query_features, jnp.float32),
            jnp.asarray(task_id, jnp.int32),
            jnp.asarray(learning_rate, jnp.float32),
        )
        return loss, grad_norm, updated

    def recall(
        self,
        query: jax.Array,
        phase: float | jax.Array,
        task_id: int | jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        self.state = self.state._replace(
            total_recalled=self.state.total_recalled
            + (self.state.count > 0).astype(jnp.int32)
        )
        return self._recall(self.state, query, phase, task_id)

    def encode(
        self,
        amplitudes: jax.Array,
        action_id: int,
        reward: float,
        task_id: int,
        phase: float,
        emotional_weight: float = 1.0,
        features: Optional[jax.Array] = None,
    ) -> None:
        self.state = self.candidate_encode(
            amplitudes, action_id, reward, task_id, phase, emotional_weight, features
        )

    def candidate_encode(
        self,
        amplitudes: jax.Array,
        action_id: int,
        reward: float,
        task_id: int,
        phase: float,
        emotional_weight: float = 1.0,
        features: Optional[jax.Array] = None,
    ) -> JAXMemoryState:
        return self._encode(
            self.state, amplitudes, action_id, reward, task_id, phase, emotional_weight,
            features,
        )

    def get_stats(self) -> dict[str, int | float | bool]:
        return {
            "total_memories": int(jax.device_get(self.state.count)),
            "total_encoded": int(jax.device_get(self.state.total_encoded)),
            "total_recalled": int(jax.device_get(self.state.total_recalled)),
            "capacity": self.capacity,
            "dim": self.dim,
        }
