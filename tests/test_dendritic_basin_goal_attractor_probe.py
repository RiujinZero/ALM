from manifoldflow.experiments.dendritic_basin_goal_attractor_probe import run_probe


def test_conditioned_goal_probe_passes_causal_gate() -> None:
    report = run_probe(episodes=80, gap=8, seed=7)
    direct = report["conditions"]["direct_only"]
    conditioned = report["conditions"]["conditioned"]

    assert direct["selected_goal"] is None
    assert conditioned["selected_goal"] == 0
    assert conditioned["restorative_candidate_value"] > 0.05
    assert conditioned["predictable_loop_value"] < 0.0
    assert conditioned["persisted_neutral_steps"] == 16
    assert conditioned["dissolved_on_restoration"]
    assert report["prediction_only_control"]["selected_goal"] == 1
    assert report["causal_goal_swap"]["changed_fraction"] == 1.0
