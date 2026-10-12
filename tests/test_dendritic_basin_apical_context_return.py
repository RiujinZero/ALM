from types import SimpleNamespace

import numpy as np

from manifoldflow.experiments import dendritic_basin_apical_context_return as apical


def test_apical_context_return_is_transient_and_receptor_positioned() -> None:
    learner = SimpleNamespace(
        temporal_bins=(1, 4, 16),
        event_history=[2, 3, 4],
        active_word_candidates={7: 2},
    )

    def predict() -> np.ndarray:
        result = np.zeros(3, dtype=np.float64)
        result[learner.event_history[-2] % 3] = 1.0
        return result

    learner.predict_next_distribution = predict
    original_history = learner.event_history.copy()
    original_candidates = learner.active_word_candidates

    result = apical._context_overlay_distribution(
        learner,
        ((8, 4),),
        support={(8, 4): 1.0},
    )

    assert int(np.argmax(result)) == 8 % 3
    assert learner.event_history == original_history
    assert learner.active_word_candidates is original_candidates
