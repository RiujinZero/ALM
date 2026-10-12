import numpy as np

from manifoldflow.dendritic_basin.shared_cortical_assemblies import (
    SharedCorticalAssemblies,
)
from manifoldflow.experiments.dendritic_basin_shared_cortical_assembly_smoke import (
    _generation_diagnostics,
)


def test_basal_and_apical_inputs_have_distinct_provenance() -> None:
    tissue = SharedCorticalAssemblies(4)
    code = np.asarray([1.0, 0.5, 0.0, 0.0])

    basal = tissue.features(code)
    apical = tissue.features(np.zeros(4), code)

    assert np.array_equal(basal[:4], apical[4:])
    assert not np.any(basal[4:])
    assert not np.any(apical[:4])


def test_external_prediction_error_recruits_and_competes() -> None:
    tissue = SharedCorticalAssemblies(4, learning_rate=1.0)
    cue = tissue.features(
        np.asarray([1.0, 0.5, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0, 0.5]),
    )

    tissue.observe_external(cue, 3)
    first = tissue.activations(cue)[3]
    tissue.observe_external(cue, 7)
    activations = tissue.activations(cue)

    assert first > 0.0
    assert activations[7] > activations.get(3, 0.0)
    assert tissue.observations == {3: 1, 7: 1}
    assert tissue.external_updates == 2


def test_zero_activity_does_not_create_synapses() -> None:
    tissue = SharedCorticalAssemblies(4)

    tissue.observe_external(tissue.features(np.zeros(4)), 5)

    assert tissue.weights[5] == {}
    assert tissue.activations(np.zeros(8)) == {}


def test_active_assembly_reconstructs_only_positive_transient_evidence() -> None:
    tissue = SharedCorticalAssemblies(2)
    tissue.weights[4] = {0: 0.5, 2: 1.0, 3: -0.75}

    reconstructed = tissue.reconstruct({4: 0.8})

    assert np.allclose(reconstructed, [0.5, 0.0, 1.0, 0.0])
    assert tissue.external_updates == 0


def test_generation_diagnostics_measure_repetition_and_encoding() -> None:
    result = {"hex": b"one one one".hex()}

    diagnostics = _generation_diagnostics(result)

    assert diagnostics["valid_utf8"] is True
    assert diagnostics["word_count"] == 3
    assert diagnostics["unique_word_fraction"] == 1 / 3
    assert diagnostics["repeated_4gram_fraction"] > 0.0
