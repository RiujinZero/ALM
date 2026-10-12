# NCCA-v1 train-only proxy-observability amendment

## Freeze status

Frozen before the first real-data execution on 14 September 2026. This
amendment authorizes only the train-only gate-zero information assay. It does
not authorize a native receiver, development/test-pair access, action
selection, writer actuation, or a goal-directedness claim.

The earlier NCCA-v1 document gated every stage on the frozen causal-antecedent
receiver. That receiver subsequently failed its free-generation development
gate. The failure retires that receiver configuration but does not make the
upstream statistical question invalid. Gate zero may therefore run separately
to ask whether logged response actions contain any held-out predictive
information about the next external USER event.

## Frozen inputs and implementation

Dataset directory:

`runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3`

Only these files may be opened:

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| `manifest.json` | 4,685 | `7999beb210e813bba37d38c903435985740de7a378d346a63c75c0fd05e705df` |
| `train.bin` | 11,999,925 | `ccb1751d350daee579a37bb9cb974e8b5cce446b958f3ccb7ea4e2ae6814c841` |
| `train.index.jsonl` | 8,834,787 | `6efb255f9bd828108f39101afcfaa950fbfcd12b8ce37956ac0c71bc7ad05e5c` |

Frozen code:

| File | SHA-256 |
| --- | --- |
| `src/manifoldflow/experiments/dendritic_basin_ncca_proxy_observability_ablation.py` | `295ea96fba54579d963b373d965dcb8d05b086e0e20f8a0c42bd2e1fc17c5280` |
| `src/manifoldflow/dendritic_basin/context_coupled_byte_writer.py` | `69c8becb081fe1615cbf6261f5f2f4ef7ca65fbc27cf69375d7d5ff1252e0ad2` |
| `tests/test_dendritic_basin_ncca_proxy_observability_ablation.py` | `ddb5fd31318923ff2155ee701e827232045c8ccd29add97bc1016149e659a642` |
| `docs/ALM_NCCA_v1_Natural_Chat_Consequence_Preregistration.md` | `331b0ac41188ed0fa9e6d80ce94e3d3c87acba2fc9ec642c8fedf748e55a56b7` |

At freeze time, 15 focused tests and 29 focused-plus-dependency tests pass;
Ruff, `py_compile`, and `git diff --check` are clean. Synthetic access
instrumentation confirms that the loader opens only the three named training
inputs even if dev/test-like decoy payloads exist.

## Exact question and endpoint

Across three conversation-disjoint rotations, fit allocation-matched `q_H`,
`q_HA`, and stratified-action-shuffle byte compressors on one fold and score the
next fold. The anonymous action tokenizer receives response bytes and fold IDs
only. It cannot receive the later USER bytes through its API.

Gate zero passes only if every rotation:

1. scores at least 30 conversations;
2. has a conversation-clustered 95% interval strictly above zero for the
   held-out BPB improvement of `q_HA` over `q_H`; and
3. has the same strict interval result for `q_HA` over the allocation-matched
   stratified action-shuffle compressor.

This BPB contrast is the object of this narrow information assay. It is not the
downstream mechanism-selection metric. A positive result means predictive
dependence only; a negative one-pass result does not reject a mechanism that
needs longer training. Independently checked free-generation learning curves
remain the primary endpoint for actuation and mechanism selection.

## Prospective singleton-stratum handling

The frozen implementation deliberately stops before compressor fitting if any
within-fit-fold `(source, script, horizon, response_length)` shuffle stratum has
fewer than two rows. Whether this occurs is a structural property, not an
outcome. No looser shuffle may be improvised after an effect estimate is seen.

If the first execution reaches this hard stop, the only permitted amendment is
the following deterministic hierarchy. For each fit fold, choose the first key
whose every group has at least two rows:

1. `(source, script, horizon, response_length)`;
2. `(source, script, horizon)`;
3. `(source, script, response_length)`;
4. `(source, script)`;
5. `(source, horizon, response_length)`;
6. `(source, horizon)`;
7. `(source, response_length)`;
8. `(source)`.

All three compressors must then train on the identical rows. Sattolo
derangement remains required inside every selected group; source is never
crossed. The report must record the selected key, group counts and sizes, row
fixed points, and effective same-action-label fraction. If source alone still
contains a singleton, or if any row fixed point remains, the assay stops. A
source amendment implementing exactly this hierarchy must be hashed and
recorded before restarting. The failed structural attempt produces no
inferential result and cannot tune any model setting.

## Execution and result boundary

The locked command is:

```powershell
.venv/Scripts/python.exe -m manifoldflow.experiments.dendritic_basin_ncca_proxy_observability_ablation `
  --dataset-dir runs/shared_goal/confirmatory_goal_action_memory_successor_lmsys_lines100000_149999_v3 `
  --report runs/shared_goal/ncca_proxy_observability_schema4_v1_train_only.json
```

The protected pair files are never opened. Do not run a protected test after
this assay, choose a receiver beta from it, or describe its compressor BPB as
free-generation quality. The next authorized work after a positive gate is a
separately frozen convergence experiment under
`docs/ALM_Shared_Goal_Convergence_Evaluation_Contract.md`.
