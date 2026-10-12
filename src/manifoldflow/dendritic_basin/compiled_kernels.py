"""Optional compiled kernels for the gradient-free dendritic learner.

The learner's causal and structural control remains explicit Python.  These
kernels compile repeated local arithmetic over bounded branch evidence.  A
pure-Python implementation is always available for portability and as a
semantic oracle; neither path imports an autograd framework.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal installations
    NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        def decorate(function):  # type: ignore[no-untyped-def]
            return function

        return decorate

    prange = range


EPS = 1e-8


def access_branch_survival_python(
    traces: np.ndarray,
    saliences: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    synapse_counts: np.ndarray,
    group_targets: np.ndarray,
    group_counts: np.ndarray,
    strengths: np.ndarray,
    last_used: np.ndarray,
    capture_saliences: np.ndarray,
    clock: int,
    false_response_grace: int,
    surprise_survival: bool,
    branch_compartments: int,
    activation_threshold: float,
    false_response_decay: float,
    survival_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordered oracle for competitive false-response branch survival."""
    strengths = strengths.copy()
    false_responses = np.zeros(len(strengths), dtype=np.int64)
    alive = np.ones(len(strengths), dtype=np.bool_)
    for trace_index in range(len(traces)):
        salience = saliences[trace_index]
        for branch_index in range(len(strengths)):
            if not alive[branch_index]:
                continue
            protected = (
                clock - last_used[branch_index] < false_response_grace
                or (
                    surprise_survival
                    and salience < capture_saliences[branch_index]
                )
            )
            response = 0.0
            response_count = 0
            product = 1.0
            if group_counts[branch_index] == 0:
                current = 0.0
                for slot in range(synapse_counts[branch_index]):
                    dimension = indices[branch_index, slot]
                    current += (
                        weights[branch_index, slot]
                        * traces[trace_index, dimension]
                    )
                response = strengths[branch_index] * min(
                    1.0, max(0.0, current)
                )
            for group in range(group_counts[branch_index]):
                target = group_targets[branch_index, group]
                if target <= 1e-12:
                    continue
                current = 0.0
                for slot in range(synapse_counts[branch_index]):
                    dimension = indices[branch_index, slot]
                    if dimension % branch_compartments == group:
                        current += (
                            weights[branch_index, slot]
                            * traces[trace_index, dimension]
                        )
                ratio = current / target
                ratio = min(1.0, max(0.0, ratio))
                product *= ratio
                response_count += 1
            if response_count:
                response = strengths[branch_index] * product ** (
                    1.0 / response_count
                )
            if not protected and response >= activation_threshold:
                strengths[branch_index] *= 1.0 - false_response_decay
                false_responses[branch_index] += 1
            if strengths[branch_index] < survival_threshold:
                alive[branch_index] = False
    return strengths, false_responses, alive


@njit(cache=True, parallel=True)
def access_branch_survival_compiled(
    traces: np.ndarray,
    saliences: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    synapse_counts: np.ndarray,
    group_targets: np.ndarray,
    group_counts: np.ndarray,
    strengths: np.ndarray,
    last_used: np.ndarray,
    capture_saliences: np.ndarray,
    clock: int,
    false_response_grace: int,
    surprise_survival: bool,
    branch_compartments: int,
    activation_threshold: float,
    false_response_decay: float,
    survival_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compiled equivalent of :func:`access_branch_survival_python`."""
    strengths = strengths.copy()
    false_responses = np.zeros(len(strengths), dtype=np.int64)
    alive = np.ones(len(strengths), dtype=np.bool_)
    for branch_index in prange(len(strengths)):
        for trace_index in range(len(traces)):
            salience = saliences[trace_index]
            if not alive[branch_index]:
                continue
            protected = (
                clock - last_used[branch_index] < false_response_grace
                or (
                    surprise_survival
                    and salience < capture_saliences[branch_index]
                )
            )
            response = 0.0
            response_count = 0
            product = 1.0
            if group_counts[branch_index] == 0:
                current = 0.0
                for slot in range(synapse_counts[branch_index]):
                    dimension = indices[branch_index, slot]
                    current += (
                        weights[branch_index, slot]
                        * traces[trace_index, dimension]
                    )
                if current < 0.0:
                    current = 0.0
                elif current > 1.0:
                    current = 1.0
                response = strengths[branch_index] * current
            for group in range(group_counts[branch_index]):
                target = group_targets[branch_index, group]
                if target <= 1e-12:
                    continue
                current = 0.0
                for slot in range(synapse_counts[branch_index]):
                    dimension = indices[branch_index, slot]
                    if dimension % branch_compartments == group:
                        current += (
                            weights[branch_index, slot]
                            * traces[trace_index, dimension]
                        )
                ratio = current / target
                if ratio < 0.0:
                    ratio = 0.0
                elif ratio > 1.0:
                    ratio = 1.0
                product *= ratio
                response_count += 1
            if response_count:
                response = strengths[branch_index] * product ** (
                    1.0 / response_count
                )
            if not protected and response >= activation_threshold:
                strengths[branch_index] *= 1.0 - false_response_decay
                false_responses[branch_index] += 1
            if strengths[branch_index] < survival_threshold:
                alive[branch_index] = False
    return strengths, false_responses, alive


def interaction_objective_python(
    samples: np.ndarray,
    w_a: float,
    w_b: float,
    mu: float,
    sigma: float,
) -> float:
    """The scalar Welford interaction objective used as the oracle."""
    n = 0
    mean_a = mean_b = mean_d = 0.0
    m2_a = m2_b = m2_d = 0.0
    cov_da_acc = cov_db_acc = 0.0
    for row in samples:
        a = float(row[0])
        b = float(row[1])
        drive = w_a * a + w_b * b
        d = math.exp(-((drive - mu) ** 2) / (2.0 * sigma**2))
        n += 1
        da_pre = a - mean_a
        db_pre = b - mean_b
        dd_pre = d - mean_d
        mean_a += da_pre / n
        mean_b += db_pre / n
        mean_d += dd_pre / n
        m2_a += da_pre * (a - mean_a)
        m2_b += db_pre * (b - mean_b)
        m2_d += dd_pre * (d - mean_d)
        cov_da_acc += da_pre * (d - mean_d)
        cov_db_acc += db_pre * (d - mean_d)
    if n == 0:
        return 0.0
    var_a = m2_a / n
    var_b = m2_b / n
    var_d = m2_d / n
    cov_d_a = cov_da_acc / n
    cov_d_b = cov_db_acc / n
    return (
        var_d
        - cov_d_a**2 / (var_a + EPS)
        - cov_d_b**2 / (var_b + EPS)
    )


@njit(cache=True)
def interaction_objective_compiled(
    samples: np.ndarray,
    w_a: float,
    w_b: float,
    mu: float,
    sigma: float,
) -> float:
    """Compiled equivalent of :func:`interaction_objective_python`."""
    n = 0
    mean_a = 0.0
    mean_b = 0.0
    mean_d = 0.0
    m2_a = 0.0
    m2_b = 0.0
    m2_d = 0.0
    cov_da_acc = 0.0
    cov_db_acc = 0.0
    for index in range(samples.shape[0]):
        a = samples[index, 0]
        b = samples[index, 1]
        drive = w_a * a + w_b * b
        d = math.exp(-((drive - mu) ** 2) / (2.0 * sigma**2))
        n += 1
        da_pre = a - mean_a
        db_pre = b - mean_b
        dd_pre = d - mean_d
        mean_a += da_pre / n
        mean_b += db_pre / n
        mean_d += dd_pre / n
        m2_a += da_pre * (a - mean_a)
        m2_b += db_pre * (b - mean_b)
        m2_d += dd_pre * (d - mean_d)
        cov_da_acc += da_pre * (d - mean_d)
        cov_db_acc += db_pre * (d - mean_d)
    if n == 0:
        return 0.0
    var_a = m2_a / n
    var_b = m2_b / n
    var_d = m2_d / n
    cov_d_a = cov_da_acc / n
    cov_d_b = cov_db_acc / n
    return (
        var_d
        - cov_d_a**2 / (var_a + EPS)
        - cov_d_b**2 / (var_b + EPS)
    )


@njit(cache=True)
def update_interaction_stats_compiled(
    slot_ids: np.ndarray,
    a_values: np.ndarray,
    b_values: np.ndarray,
    responses: np.ndarray,
    stats_n: np.ndarray,
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    mean_d: np.ndarray,
    m2_a: np.ndarray,
    m2_b: np.ndarray,
    m2_d: np.ndarray,
    cov_da_acc: np.ndarray,
    cov_db_acc: np.ndarray,
) -> None:
    """Accumulate branch-local Welford evidence without Python objects."""
    for raw_slot_id in slot_ids:
        slot_id = int(raw_slot_id)
        a = a_values[slot_id]
        b = b_values[slot_id]
        d = responses[slot_id]
        n = stats_n[slot_id] + 1
        da_pre = a - mean_a[slot_id]
        db_pre = b - mean_b[slot_id]
        dd_pre = d - mean_d[slot_id]
        mean_a[slot_id] += da_pre / n
        mean_b[slot_id] += db_pre / n
        mean_d[slot_id] += dd_pre / n
        m2_a[slot_id] += da_pre * (a - mean_a[slot_id])
        m2_b[slot_id] += db_pre * (b - mean_b[slot_id])
        m2_d[slot_id] += dd_pre * (d - mean_d[slot_id])
        cov_da_acc[slot_id] += da_pre * (d - mean_d[slot_id])
        cov_db_acc[slot_id] += db_pre * (d - mean_d[slot_id])
        stats_n[slot_id] = n


@njit(cache=True)
def _ring_interaction_objective_compiled(
    samples: np.ndarray,
    count: int,
    position: int,
    w_a: float,
    w_b: float,
    mu: float,
    sigma: float,
) -> float:
    n = 0
    mean_a = 0.0
    mean_b = 0.0
    mean_d = 0.0
    m2_a = 0.0
    m2_b = 0.0
    m2_d = 0.0
    cov_da_acc = 0.0
    cov_db_acc = 0.0
    capacity = samples.shape[0]
    start = position if count == capacity else 0
    for offset in range(count):
        index = (start + offset) % capacity
        a = samples[index, 0]
        b = samples[index, 1]
        drive = w_a * a + w_b * b
        d = math.exp(-((drive - mu) ** 2) / (2.0 * sigma**2))
        n += 1
        da_pre = a - mean_a
        db_pre = b - mean_b
        dd_pre = d - mean_d
        mean_a += da_pre / n
        mean_b += db_pre / n
        mean_d += dd_pre / n
        m2_a += da_pre * (a - mean_a)
        m2_b += db_pre * (b - mean_b)
        m2_d += dd_pre * (d - mean_d)
        cov_da_acc += da_pre * (d - mean_d)
        cov_db_acc += db_pre * (d - mean_d)
    if n == 0:
        return 0.0
    var_a = m2_a / n
    var_b = m2_b / n
    var_d = m2_d / n
    cov_d_a = cov_da_acc / n
    cov_d_b = cov_db_acc / n
    return (
        var_d
        - cov_d_a**2 / (var_a + EPS)
        - cov_d_b**2 / (var_b + EPS)
    )


@njit(cache=True)
def adapt_slots_compiled(
    slot_ids: np.ndarray,
    perturbations: np.ndarray,
    local_samples: np.ndarray,
    local_sample_counts: np.ndarray,
    local_sample_positions: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    minimum_sigma: float,
) -> None:
    """Perturb/select a set of local dendrites in anatomical RNG order."""
    for row in range(slot_ids.shape[0]):
        slot_id = int(slot_ids[row])
        candidate_w_a = w_a[slot_id] + perturbations[row, 0]
        candidate_w_b = w_b[slot_id] + perturbations[row, 1]
        candidate_mu = mu[slot_id] + perturbations[row, 2]
        candidate_sigma = max(
            1e-3, sigma[slot_id] + perturbations[row, 3]
        )
        count = int(local_sample_counts[slot_id])
        position = int(local_sample_positions[slot_id])
        current_j = _ring_interaction_objective_compiled(
            local_samples[slot_id],
            count,
            position,
            w_a[slot_id],
            w_b[slot_id],
            mu[slot_id],
            sigma[slot_id],
        )
        candidate_j = _ring_interaction_objective_compiled(
            local_samples[slot_id],
            count,
            position,
            candidate_w_a,
            candidate_w_b,
            candidate_mu,
            candidate_sigma,
        )
        if candidate_j > current_j:
            w_a[slot_id] = candidate_w_a
            w_b[slot_id] = candidate_w_b
            mu[slot_id] = candidate_mu
            sigma[slot_id] = candidate_sigma
        sigma[slot_id] = max(minimum_sigma, sigma[slot_id])


@njit(cache=True)
def _splitmix64(value: np.uint64) -> np.uint64:
    value += np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    value = (value ^ (value >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return value ^ (value >> np.uint64(31))


@njit(cache=True)
def counter_perturbations_compiled(
    slot_ids: np.ndarray,
    observations: np.ndarray,
    seed: int,
    perturb_step: float,
) -> np.ndarray:
    """Generate stateless branch-local Gaussian proposals.

    The key is anatomical identity plus local observation count. This makes
    local-search noise independent of execution batching while retaining the
    structural seed as the organism-level developmental identity.
    """
    result = np.empty((slot_ids.shape[0], 4))
    inverse_53 = 1.0 / 9007199254740992.0
    for row in range(slot_ids.shape[0]):
        slot_id = int(slot_ids[row])
        base = (
            np.uint64(seed)
            ^ (np.uint64(slot_id + 1) * np.uint64(0xD2B74407B1CE6E93))
            ^ (
                np.uint64(observations[slot_id] + 1)
                * np.uint64(0xCA5A826395121157)
            )
        )
        for pair in range(2):
            first = _splitmix64(base ^ np.uint64(2 * pair + 1))
            second = _splitmix64(base ^ np.uint64(2 * pair + 2))
            u1 = ((first >> np.uint64(11)) + 0.5) * inverse_53
            u2 = ((second >> np.uint64(11)) + 0.5) * inverse_53
            radius = math.sqrt(-2.0 * math.log(u1)) * perturb_step
            angle = 2.0 * math.pi * u2
            result[row, 2 * pair] = radius * math.cos(angle)
            result[row, 2 * pair + 1] = radius * math.sin(angle)
    return result


@njit(cache=True)
def select_column_winners_compiled(
    weights: np.ndarray,
    slot_ids: np.ndarray,
    candidate_positions: np.ndarray,
    column_size: int,
    column_count: int,
    max_per_column: int,
    max_columns: int,
    max_active: int,
) -> tuple[np.ndarray, int]:
    """Stable sparse tournament: branch -> column -> soma winners."""
    local_positions = np.full((column_count, max_per_column), -1, np.int64)
    local_weights = np.full((column_count, max_per_column), -np.inf)
    for raw_position in candidate_positions:
        position = int(raw_position)
        column = int(slot_ids[position]) // column_size
        weight = weights[position]
        for rank in range(max_per_column):
            if weight > local_weights[column, rank]:
                for shift in range(max_per_column - 1, rank, -1):
                    local_weights[column, shift] = local_weights[column, shift - 1]
                    local_positions[column, shift] = local_positions[
                        column, shift - 1
                    ]
                local_weights[column, rank] = weight
                local_positions[column, rank] = position
                break

    column_scores = np.zeros(column_count)
    for column in range(column_count):
        for rank in range(max_per_column):
            if local_positions[column, rank] >= 0:
                column_scores[column] += local_weights[column, rank]

    active_columns = np.zeros(column_count, np.bool_)
    for _rank in range(min(max_columns, column_count)):
        best_column = -1
        best_score = -np.inf
        for column in range(column_count):
            if not active_columns[column] and column_scores[column] > best_score:
                best_score = column_scores[column]
                best_column = column
        if best_column < 0 or best_score <= 0.0:
            break
        active_columns[best_column] = True

    result = np.full(max_active, -1, np.int64)
    result_weights = np.full(max_active, -np.inf)
    count = 0
    for column in range(column_count):
        if not active_columns[column]:
            continue
        for local_rank in range(max_per_column):
            position = local_positions[column, local_rank]
            if position < 0:
                continue
            weight = weights[position]
            if count >= max_active and weight <= result_weights[max_active - 1]:
                continue
            insertion = min(count, max_active - 1)
            while insertion > 0 and weight > result_weights[insertion - 1]:
                if insertion < max_active:
                    result[insertion] = result[insertion - 1]
                    result_weights[insertion] = result_weights[insertion - 1]
                insertion -= 1
            if insertion < max_active:
                result[insertion] = position
                result_weights[insertion] = weight
            count = min(count + 1, max_active)
    return result, count


@njit(cache=True)
def compete_predictors_compiled(
    evaluated_ids: np.ndarray,
    features: np.ndarray,
    raw_responses: np.ndarray,
    state_probes: np.ndarray,
    live: np.ndarray,
    lifecycle: np.ndarray,
    observations: np.ndarray,
    fires: np.ndarray,
    utility_bits: np.ndarray,
    state_observations: np.ndarray,
    state_utility_bits: np.ndarray,
    predictor_counts: np.ndarray,
    state_predictor_counts: np.ndarray,
    base_counts: np.ndarray,
    context_counts: np.ndarray,
    previous_token: int,
    contextual_residual: bool,
    vocab_size: int,
    dormant_lifecycle: int,
    activation_threshold: float,
    relation_prior: float,
    base_prior: float,
    context_prior: float,
    column_size: int,
    column_count: int,
    max_per_column: int,
    max_columns: int,
    max_active: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, int]:
    """Run branch evidence, column competition, and predictor mixing together.

    The returned local relation set contains every event/state branch that is
    entitled to learn the next causal outcome, including branches that lose
    soma-level competition.  Only winner distributions are normalized.
    """
    local_ids = np.empty(evaluated_ids.shape[0], np.int64)
    local_is_state = np.empty(evaluated_ids.shape[0], np.bool_)
    local_weights = np.zeros(evaluated_ids.shape[0])
    candidate_positions = np.empty(evaluated_ids.shape[0], np.int64)
    local_count = 0
    candidate_count = 0

    for raw_slot_id in evaluated_ids:
        slot_id = int(raw_slot_id)
        if (
            not live[slot_id]
            or lifecycle[slot_id] == dormant_lifecycle
            or observations[slot_id] < 2
            or fires[slot_id] < 1
        ):
            continue
        event_active = raw_responses[slot_id] >= activation_threshold
        state_active = (
            not event_active and state_probes[slot_id] >= activation_threshold
        )
        if not event_active and not state_active:
            continue

        position = local_count
        local_ids[position] = slot_id
        local_is_state[position] = state_active
        local_count += 1

        activation = features[vocab_size + slot_id]
        if activation < activation_threshold:
            continue
        if state_active:
            evidence = state_observations[slot_id] / (
                state_observations[slot_id] + 16.0
            )
            utility = state_utility_bits[slot_id]
        else:
            evidence = observations[slot_id] / (observations[slot_id] + 8.0)
            utility = utility_bits[slot_id]
        if fires[slot_id] < 8:
            utility_factor = 0.10
        elif utility > 0.0:
            utility_factor = min(4.0, 0.25 + 4.0 * utility)
        else:
            utility_factor = 0.0
        weight = activation * evidence * utility_factor
        local_weights[position] = weight
        if weight > 0.0:
            candidate_positions[candidate_count] = position
            candidate_count += 1

    basal_prior = context_prior if contextual_residual else base_prior
    base_total = basal_prior * vocab_size
    for token_id in range(vocab_size):
        if contextual_residual and previous_token >= 0:
            base_total += context_counts[previous_token, token_id]
        else:
            base_total += base_counts[token_id]
    mixture = np.empty(vocab_size)
    for token_id in range(vocab_size):
        count = base_counts[token_id]
        if contextual_residual and previous_token >= 0:
            count = context_counts[previous_token, token_id]
        mixture[token_id] = (count + basal_prior) / base_total

    if contextual_residual:
        for token_id in range(vocab_size):
            mixture[token_id] = math.log(max(mixture[token_id], 1e-300))

    winner_positions = np.full(max_active, -1, np.int64)
    winner_count = 0
    if candidate_count > 0:
        candidates = candidate_positions[:candidate_count]
        if column_count == 1:
            winner_weights = np.full(max_active, -np.inf)
            for raw_position in candidates:
                position = int(raw_position)
                weight = local_weights[position]
                if (
                    winner_count >= max_active
                    and weight <= winner_weights[max_active - 1]
                ):
                    continue
                insertion = min(winner_count, max_active - 1)
                while insertion > 0 and weight > winner_weights[insertion - 1]:
                    winner_positions[insertion] = winner_positions[insertion - 1]
                    winner_weights[insertion] = winner_weights[insertion - 1]
                    insertion -= 1
                winner_positions[insertion] = position
                winner_weights[insertion] = weight
                winner_count = min(winner_count + 1, max_active)
        else:
            winner_positions, winner_count = select_column_winners_compiled(
                local_weights,
                local_ids,
                candidates,
                column_size,
                column_count,
                max_per_column,
                max_columns,
                max_active,
            )

    total_weight = 1.0
    for winner_index in range(winner_count):
        position = int(winner_positions[winner_index])
        slot_id = int(local_ids[position])
        weight = local_weights[position]
        count_total = relation_prior * vocab_size
        if local_is_state[position]:
            for token_id in range(vocab_size):
                count_total += state_predictor_counts[slot_id, token_id]
            for token_id in range(vocab_size):
                probability = (
                    state_predictor_counts[slot_id, token_id] + relation_prior
                ) / count_total
                if contextual_residual:
                    mixture[token_id] += weight * math.log(max(probability, 1e-300))
                else:
                    mixture[token_id] += weight * probability
        else:
            for token_id in range(vocab_size):
                count_total += predictor_counts[slot_id, token_id]
            for token_id in range(vocab_size):
                probability = (
                    predictor_counts[slot_id, token_id] + relation_prior
                ) / count_total
                if contextual_residual:
                    mixture[token_id] += weight * math.log(max(probability, 1e-300))
                else:
                    mixture[token_id] += weight * probability
        total_weight += weight
    if contextual_residual:
        maximum = -np.inf
        for token_id in range(vocab_size):
            mixture[token_id] /= total_weight
            maximum = max(maximum, mixture[token_id])
        probability_total = 0.0
        for token_id in range(vocab_size):
            mixture[token_id] = math.exp(mixture[token_id] - maximum)
            probability_total += mixture[token_id]
        for token_id in range(vocab_size):
            mixture[token_id] /= probability_total
    else:
        for token_id in range(vocab_size):
            mixture[token_id] /= total_weight

    winner_ids = np.empty(winner_count, np.int64)
    for winner_index in range(winner_count):
        winner_ids[winner_index] = local_ids[winner_positions[winner_index]]
    return (
        mixture,
        winner_ids,
        winner_count,
        local_ids,
        local_is_state,
        local_count,
    )


@njit(cache=True)
def update_local_outcomes_compiled(
    relation_ids: np.ndarray,
    relation_is_state: np.ndarray,
    token_id: int,
    base_probability: float,
    relation_prior: float,
    utility_decay: float,
    predictor_decay: float,
    consolidated_predictor_decay: float,
    consolidated_lifecycle: int,
    lifecycle: np.ndarray,
    predictor_counts: np.ndarray,
    state_predictor_counts: np.ndarray,
    utility_bits: np.ndarray,
    state_utility_bits: np.ndarray,
    state_observations: np.ndarray,
) -> None:
    """Resolve one causal outcome for every locally active branch."""
    safe_base = max(base_probability, 1e-12)
    vocab_size = predictor_counts.shape[1]
    for position in range(relation_ids.shape[0]):
        slot_id = int(relation_ids[position])
        is_state = relation_is_state[position]
        total = relation_prior * vocab_size
        if is_state:
            for token in range(vocab_size):
                total += state_predictor_counts[slot_id, token]
            probability = (
                state_predictor_counts[slot_id, token_id] + relation_prior
            ) / total
        else:
            for token in range(vocab_size):
                total += predictor_counts[slot_id, token]
            probability = (
                predictor_counts[slot_id, token_id] + relation_prior
            ) / total
        gain = math.log2(max(probability, 1e-12) / safe_base)
        decay = predictor_decay
        if lifecycle[slot_id] == consolidated_lifecycle:
            decay = consolidated_predictor_decay
        if is_state:
            state_utility_bits[slot_id] = (
                utility_decay * state_utility_bits[slot_id]
                + (1.0 - utility_decay) * gain
            )
            for token in range(vocab_size):
                state_predictor_counts[slot_id, token] *= decay
            state_predictor_counts[slot_id, token_id] += 1.0
            state_observations[slot_id] += 1
        else:
            utility_bits[slot_id] = (
                utility_decay * utility_bits[slot_id]
                + (1.0 - utility_decay) * gain
            )
            for token in range(vocab_size):
                predictor_counts[slot_id, token] *= decay
            predictor_counts[slot_id, token_id] += 1.0


@njit(cache=True)
def advance_response_batch_compiled(
    slot_ids: np.ndarray,
    current_channels: np.ndarray,
    trace_channels: np.ndarray,
    timescale_indices: np.ndarray,
    features: np.ndarray,
    trace_scales: np.ndarray,
    traces: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    persistence: np.ndarray,
    state_observations: np.ndarray,
    state_utility_bits: np.ndarray,
    vocab_size: int,
    support_threshold: float,
    activation_threshold: float,
    persistence_decay: float,
    persistence_min_observations: int,
    persistence_utility_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance one independent causal-depth stratum in place."""
    count = slot_ids.shape[0]
    a_values = np.zeros(count)
    b_values = np.zeros(count)
    responses = np.zeros(count)
    state_probes = np.zeros(count)
    for position in range(count):
        slot_id = int(slot_ids[position])
        a = features[int(current_channels[position])]
        timescale_index = int(timescale_indices[position])
        b = trace_scales[timescale_index] * traces[
            timescale_index, int(trace_channels[position])
        ]
        b = min(1.0, b)
        response = 0.0
        if a > support_threshold:
            drive = w_a[slot_id] * a + w_b[slot_id] * b
            response = math.exp(
                -((drive - mu[slot_id]) ** 2) / (2.0 * sigma[slot_id] ** 2)
            )
        a_values[position] = a
        b_values[position] = b
        responses[position] = response

        event_active = response >= activation_threshold
        retained = persistence[slot_id] * persistence_decay
        if event_active:
            retained = response
        persistence[slot_id] = retained
        if not event_active and retained >= activation_threshold:
            state_probes[position] = retained

        persistent_activation = 0.0
        observations = state_observations[slot_id]
        if (
            observations >= persistence_min_observations
            and state_utility_bits[slot_id] >= persistence_utility_threshold
        ):
            confidence = observations / (observations + 16.0)
            persistent_activation = retained * confidence
        activation = max(response, persistent_activation)
        relation_channel = vocab_size + slot_id
        features[relation_channel] = activation
        for trace_index in range(traces.shape[0]):
            traces[trace_index, relation_channel] += activation
    return a_values, b_values, responses, state_probes


@njit(cache=True)
def advance_dense_features_compiled(
    token_id: int,
    live_ids: np.ndarray,
    current_channels: np.ndarray,
    trace_channels: np.ndarray,
    timescale_indices: np.ndarray,
    lambdas: np.ndarray,
    trace_scales: np.ndarray,
    traces: np.ndarray,
    trace_means: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    persistence: np.ndarray,
    state_observations: np.ndarray,
    state_utility_bits: np.ndarray,
    vocab_size: int,
    max_relations: int,
    support_threshold: float,
    activation_threshold: float,
    persistence_decay: float,
    persistence_min_observations: int,
    persistence_utility_threshold: float,
    trace_mean_decay: float,
    learn: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Advance one complete topologically ordered dense cortical step.

    Structural growth guarantees that every relation afferent has a lower
    slot ID than its consumer.  Anatomical order is therefore a valid causal
    topological order and removes the need to cross the Python boundary once
    per depth stratum.
    """
    channel_count = traces.shape[1]
    features = np.zeros(channel_count)
    raw_responses = np.zeros(max_relations)
    state_probes = np.zeros(max_relations)
    current_values = np.zeros(max_relations)
    trace_values = np.zeros(max_relations)
    features[token_id] = 1.0

    for trace_index in range(traces.shape[0]):
        decay = lambdas[trace_index]
        for channel in range(channel_count):
            traces[trace_index, channel] *= decay
        traces[trace_index, token_id] += 1.0

    for raw_slot_id in live_ids:
        slot_id = int(raw_slot_id)
        current_channel = int(current_channels[slot_id])
        trace_channel = int(trace_channels[slot_id])
        timescale_index = int(timescale_indices[slot_id])
        a = features[current_channel]
        b = min(
            1.0,
            trace_scales[timescale_index]
            * traces[timescale_index, trace_channel],
        )
        response = 0.0
        if a > support_threshold:
            drive = w_a[slot_id] * a + w_b[slot_id] * b
            response = math.exp(
                -((drive - mu[slot_id]) ** 2) / (2.0 * sigma[slot_id] ** 2)
            )
        current_values[slot_id] = a
        trace_values[slot_id] = b
        raw_responses[slot_id] = response

        event_active = response >= activation_threshold
        retained = persistence[slot_id] * persistence_decay
        if event_active:
            retained = response
        persistence[slot_id] = retained
        if not event_active and retained >= activation_threshold:
            state_probes[slot_id] = retained

        persistent_activation = 0.0
        observations = state_observations[slot_id]
        if (
            observations >= persistence_min_observations
            and state_utility_bits[slot_id] >= persistence_utility_threshold
        ):
            confidence = observations / (observations + 16.0)
            persistent_activation = retained * confidence
        activation = max(response, persistent_activation)
        relation_channel = vocab_size + slot_id
        features[relation_channel] = activation
        for trace_index in range(traces.shape[0]):
            traces[trace_index, relation_channel] += activation

    normalized = np.empty_like(traces)
    for trace_index in range(traces.shape[0]):
        scale = trace_scales[trace_index]
        for channel in range(channel_count):
            value = min(1.0, max(0.0, scale * traces[trace_index, channel]))
            normalized[trace_index, channel] = value
            if learn:
                trace_means[trace_index, channel] = (
                    trace_mean_decay * trace_means[trace_index, channel]
                    + (1.0 - trace_mean_decay) * value
                )
    return (
        features,
        normalized,
        raw_responses,
        state_probes,
        current_values,
        trace_values,
    )


@njit(cache=True)
def advance_dense_chunk_compiled(
    tokens: np.ndarray,
    has_pending: bool,
    pending_distribution: np.ndarray,
    pending_base_distribution: np.ndarray,
    pending_ids: np.ndarray,
    pending_is_state: np.ndarray,
    pending_features: np.ndarray,
    pending_traces: np.ndarray,
    live_ids: np.ndarray,
    live: np.ndarray,
    current_channels: np.ndarray,
    trace_channels: np.ndarray,
    timescale_indices: np.ndarray,
    born_steps: np.ndarray,
    last_fire_steps: np.ndarray,
    lifecycle: np.ndarray,
    consolidated_steps: np.ndarray,
    dormant_steps: np.ndarray,
    observations: np.ndarray,
    fires: np.ndarray,
    utility_bits: np.ndarray,
    state_observations: np.ndarray,
    state_utility_bits: np.ndarray,
    predictor_counts: np.ndarray,
    state_predictor_counts: np.ndarray,
    base_counts: np.ndarray,
    context_counts: np.ndarray,
    previous_token: int,
    contextual_residual: bool,
    lambdas: np.ndarray,
    trace_scales: np.ndarray,
    traces: np.ndarray,
    trace_means: np.ndarray,
    persistence: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    local_samples: np.ndarray,
    local_sample_counts: np.ndarray,
    local_sample_positions: np.ndarray,
    stats_n: np.ndarray,
    stats_mean_a: np.ndarray,
    stats_mean_b: np.ndarray,
    stats_mean_d: np.ndarray,
    stats_m2_a: np.ndarray,
    stats_m2_b: np.ndarray,
    stats_m2_d: np.ndarray,
    stats_cov_da_acc: np.ndarray,
    stats_cov_db_acc: np.ndarray,
    steps: int,
    last_birth_step: int,
    seed: int,
    vocab_size: int,
    max_relations: int,
    juvenile_lifecycle: int,
    consolidated_lifecycle: int,
    dormant_lifecycle: int,
    support_threshold: float,
    activation_threshold: float,
    persistence_decay: float,
    persistence_min_observations: int,
    persistence_utility_threshold: float,
    trace_mean_decay: float,
    base_decay: float,
    context_decay: float,
    base_prior: float,
    context_prior: float,
    relation_prior: float,
    utility_decay: float,
    predictor_decay: float,
    consolidated_predictor_decay: float,
    consolidation_min_age: int,
    consolidation_min_fires: int,
    consolidation_utility_threshold: float,
    dormancy_staleness: int,
    prune_utility: float,
    adapt_interval: int,
    consolidated_adapt_interval: int,
    min_adapt_samples: int,
    perturb_step: float,
    minimum_sigma: float,
    surprise_birth_bits: float,
    birth_cooldown: int,
    column_size: int,
    column_count: int,
    max_per_column: int,
    max_columns: int,
    max_active: int,
    learn: bool,
) -> tuple:
    """Advance ordinary causal bytes until a Python structural birth barrier."""
    observed_probabilities = np.full(tokens.shape[0], np.nan)
    base_observed_probabilities = np.full(tokens.shape[0], np.nan)
    processed = 0
    barrier = 0
    scored_delta = 0
    branch_evaluations_delta = 0
    firings_delta = 0
    consolidations_delta = 0
    dormancies_delta = 0
    reactivations_delta = 0
    winner_ids = np.empty(0, np.int64)

    for token_position in range(tokens.shape[0]):
        token_id = int(tokens[token_position])
        if has_pending:
            observed_probability = pending_distribution[token_id]
            base_probability = pending_base_distribution[token_id]
            surprise = -math.log2(max(observed_probability, 1e-12))
            if (
                learn
                and surprise >= surprise_birth_bits
                and steps - last_birth_step >= birth_cooldown
            ):
                barrier = 1
                break
            observed_probabilities[processed] = observed_probability
            base_observed_probabilities[processed] = base_probability
            if learn:
                if pending_ids.shape[0] > 0:
                    update_local_outcomes_compiled(
                        pending_ids,
                        pending_is_state,
                        token_id,
                        base_probability,
                        relation_prior,
                        utility_decay,
                        predictor_decay,
                        consolidated_predictor_decay,
                        consolidated_lifecycle,
                        lifecycle,
                        predictor_counts,
                        state_predictor_counts,
                        utility_bits,
                        state_utility_bits,
                        state_observations,
                    )
                scored_delta += 1

        if learn:
            for byte_id in range(vocab_size):
                base_counts[byte_id] *= base_decay
            base_counts[token_id] += 1.0
            if contextual_residual and previous_token >= 0:
                for byte_id in range(vocab_size):
                    context_counts[previous_token, byte_id] *= context_decay
                context_counts[previous_token, token_id] += 1.0
        previous_token = token_id

        (
            features,
            normalized_traces,
            raw_responses,
            state_probes,
            current_values,
            trace_values,
        ) = advance_dense_features_compiled(
            token_id,
            live_ids,
            current_channels,
            trace_channels,
            timescale_indices,
            lambdas,
            trace_scales,
            traces,
            trace_means,
            w_a,
            w_b,
            mu,
            sigma,
            persistence,
            state_observations,
            state_utility_bits,
            vocab_size,
            max_relations,
            support_threshold,
            activation_threshold,
            persistence_decay,
            persistence_min_observations,
            persistence_utility_threshold,
            trace_mean_decay,
            learn,
        )

        if learn:
            branch_evaluations_delta += live_ids.shape[0]
            firing_ids = np.empty(live_ids.shape[0], np.int64)
            firing_count = 0
            for raw_slot_id in live_ids:
                slot_id = int(raw_slot_id)
                if raw_responses[slot_id] >= activation_threshold:
                    firing_ids[firing_count] = slot_id
                    firing_count += 1
                    last_fire_steps[slot_id] = steps
                    if (
                        lifecycle[slot_id] == dormant_lifecycle
                        and utility_bits[slot_id]
                        >= consolidation_utility_threshold
                    ):
                        lifecycle[slot_id] = consolidated_lifecycle
                        dormant_steps[slot_id] = -1
                        reactivations_delta += 1

            for raw_slot_id in live_ids:
                slot_id = int(raw_slot_id)
                if (
                    lifecycle[slot_id] == juvenile_lifecycle
                    and steps - born_steps[slot_id] >= consolidation_min_age
                    and fires[slot_id] >= consolidation_min_fires
                    and utility_bits[slot_id]
                    >= consolidation_utility_threshold
                ):
                    lifecycle[slot_id] = consolidated_lifecycle
                    consolidated_steps[slot_id] = steps
                    consolidations_delta += 1
                if (
                    lifecycle[slot_id] == consolidated_lifecycle
                    and (
                        steps - last_fire_steps[slot_id] >= dormancy_staleness
                        or utility_bits[slot_id] < prune_utility
                    )
                ):
                    lifecycle[slot_id] = dormant_lifecycle
                    dormant_steps[slot_id] = steps
                    dormancies_delta += 1

            plastic_ids = np.empty(live_ids.shape[0], np.int64)
            supported_ids = np.empty(live_ids.shape[0], np.int64)
            plastic_count = 0
            supported_count = 0
            for raw_slot_id in live_ids:
                slot_id = int(raw_slot_id)
                if lifecycle[slot_id] == dormant_lifecycle:
                    continue
                plastic_ids[plastic_count] = slot_id
                plastic_count += 1
                position = int(local_sample_positions[slot_id])
                local_samples[slot_id, position, 0] = current_values[slot_id]
                local_samples[slot_id, position, 1] = trace_values[slot_id]
                local_sample_positions[slot_id] = (
                    position + 1
                ) % local_samples.shape[1]
                local_sample_counts[slot_id] = min(
                    local_samples.shape[1], local_sample_counts[slot_id] + 1
                )
                observations[slot_id] += 1
                if current_values[slot_id] > support_threshold:
                    supported_ids[supported_count] = slot_id
                    supported_count += 1

            update_interaction_stats_compiled(
                supported_ids[:supported_count],
                current_values,
                trace_values,
                raw_responses,
                stats_n,
                stats_mean_a,
                stats_mean_b,
                stats_mean_d,
                stats_m2_a,
                stats_m2_b,
                stats_m2_d,
                stats_cov_da_acc,
                stats_cov_db_acc,
            )
            for index in range(firing_count):
                fires[firing_ids[index]] += 1
            firings_delta += firing_count

            adapting_ids = np.empty(plastic_count, np.int64)
            adapting_count = 0
            for index in range(plastic_count):
                slot_id = int(plastic_ids[index])
                interval = adapt_interval
                if lifecycle[slot_id] == consolidated_lifecycle:
                    interval = consolidated_adapt_interval
                if (
                    observations[slot_id] % interval == 0
                    and local_sample_counts[slot_id] >= min_adapt_samples
                ):
                    adapting_ids[adapting_count] = slot_id
                    adapting_count += 1
            if adapting_count:
                adapting = adapting_ids[:adapting_count]
                perturbations = counter_perturbations_compiled(
                    adapting, observations, seed, perturb_step
                )
                adapt_slots_compiled(
                    adapting,
                    perturbations,
                    local_samples,
                    local_sample_counts,
                    local_sample_positions,
                    w_a,
                    w_b,
                    mu,
                    sigma,
                    minimum_sigma,
                )

        (
            pending_distribution,
            winner_ids,
            _winner_count,
            local_ids,
            local_is_state,
            local_count,
        ) = compete_predictors_compiled(
            live_ids,
            features,
            raw_responses,
            state_probes,
            live,
            lifecycle,
            observations,
            fires,
            utility_bits,
            state_observations,
            state_utility_bits,
            predictor_counts,
            state_predictor_counts,
            base_counts,
            context_counts,
            previous_token,
            contextual_residual,
            vocab_size,
            dormant_lifecycle,
            activation_threshold,
            relation_prior,
            base_prior,
            context_prior,
            column_size,
            column_count,
            max_per_column,
            max_columns,
            max_active,
        )
        pending_ids = local_ids[:local_count].copy()
        pending_is_state = local_is_state[:local_count].copy()
        basal_prior = context_prior if contextual_residual else base_prior
        base_total = basal_prior * vocab_size
        for byte_id in range(vocab_size):
            if contextual_residual:
                base_total += context_counts[previous_token, byte_id]
            else:
                base_total += base_counts[byte_id]
        pending_base_distribution = np.empty(vocab_size)
        for byte_id in range(vocab_size):
            count = base_counts[byte_id]
            if contextual_residual:
                count = context_counts[previous_token, byte_id]
            pending_base_distribution[byte_id] = (
                count + basal_prior
            ) / base_total
        pending_features = features
        pending_traces = normalized_traces
        has_pending = True
        if learn:
            steps += 1
        processed += 1

    return (
        processed,
        barrier,
        observed_probabilities[:processed],
        base_observed_probabilities[:processed],
        has_pending,
        pending_distribution,
        pending_base_distribution,
        pending_ids,
        pending_is_state,
        winner_ids,
        pending_features,
        pending_traces,
        previous_token,
        steps,
        scored_delta,
        branch_evaluations_delta,
        firings_delta,
        consolidations_delta,
        dormancies_delta,
        reactivations_delta,
    )
