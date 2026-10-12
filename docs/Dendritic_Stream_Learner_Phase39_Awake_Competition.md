# Phase 39: Awake Dendritic Competition

## Result

The prediction-error motif cortex completed a 300,000,000-byte TinyStories
developmental gate at **1.155120872 bits per byte (bpb)** on the frozen first
1,000,000 validation bytes. Its mature embodied size is **151,382,898 bytes**.

This is the first exposure/body saturation result:

| Training exposure | Validation bpb | Mature body |
|---:|---:|---:|
| 30MB, original | 1.226226 | 58,983,428 B |
| 100MB, original | 1.178840 | 146,735,444 B |
| 30MB, awake competition | **1.224217** | **36,084,079 B** |
| 300MB, awake competition | **1.155121** | **151,382,898 B** |

Three times the previous maximum exposure added only 4.65MB over the old
100MB body while improving validation by 0.02372 bpb. Anatomy is therefore
growing sublinearly rather than tracking every observed byte.

## Why lossless packing was rejected

A lossless contiguous-array prototype emitted bit-identical probabilities.
On the 1MB gate, full packing reduced the body from 5.179MB to 4.427MB, but
frozen evaluation fell from about 120KB/s to 9.9KB/s. Restricting packing to
the slow event layer retained about 60.5KB/s, but saved only 176KB after motif
IDs were optimally renumbered.

That attacked Python representation overhead, not learned redundancy. The
prototype was removed rather than adding a slow archival anatomy to the hot
path.

## Awake homeostasis

Each sequence and event depth now owns a bounded candidate nursery. Novel
dendrites enter that local pool. Repeated candidates reach the existing birth
threshold and become stable branches; when the pool is full, the oldest still
immature candidate yields its slot. This is continuous local competition, not
a scheduled global sleep cycle.

The default capacity is 50,000 immature paths per dendritic depth. Mature
branches remain sparse dictionaries, preserving the fast lookup path. The
capacities are part of the cortex configuration and must match when
territories consolidate.

Sleep remains a rare fallback. At the end of a bounded experiment it removes
the residual nursery, then mature event branches with fewer than three global
observations lose a final use-dependent competition.

## Generality gates after competition

The same threshold-three mature competition was applied without domain-specific
rules:

| Domain | Sequence target bits | Motif target bits | Gain |
|---|---:|---:|---:|
| Arbitrary delayed symbols | 1.000000 | 0.485534 | **0.514466** |
| Occluded ring trajectory | 3.000754 | 2.651292 | **0.349462** |
| Time-warped audio motif | 1.001429 | 0.828521 | **0.172908** |
| Spider-style cross-modal cue/action | 1.000589 | 0.749638 | **0.250950** |

All four retain a gain over the depth-six sequence cortex. Cross-modal
prediction improves after weak branches are removed.

## 30MB structural calibration

Thirty 1MB territories followed by final branch competition produced:

- 1.224706 bpb;
- 50,498,275 embodied bytes;
- 74,925 validation bytes/s;
- 271,728 pruned event branches and 512,268 pruned synapses.

Three bounded 10MB awake territories produced:

- **1.224217 bpb**;
- **36,084,079 embodied bytes**;
- 73,971 validation bytes/s;
- 1,721,314 online sequence-nursery displacements;
- 3,081,308 online event-nursery displacements.

The larger territories improve within-column continuity while the bounded
nurseries reduce the body by another 28.5% relative to end-only competition.

## 300MB envelope

Thirty independent 10MB territories ran concurrently and then aligned their
assemblies during consolidation:

- training: 725.63 seconds, 413,432 bytes/s;
- consolidation and sleep: 69.30 seconds;
- validation: **1.155120872 bpb**, 36,330 bytes/s;
- checkpoint: 151,387,718 bytes;
- motifs: 1,101,708;
- event branches: 571,872;
- event synapses: 15,895,869;
- sequence branches: 593,599;
- sequence synapses: 2,674,721.

Awake competition displaced 17,504,843 immature sequence paths and 31,296,197
immature event paths. Peak aggregate Python working set stabilized near 14GB
instead of exhausting the 32GB host, and the completed nursery is empty.

Thirty simultaneous processes were slower than the three-process calibration
because the pointer-heavy anatomy saturated cache and memory bandwidth. This
does not require reducing developmental diversity: future runs can retain
thirty 10MB territories while scheduling only 8--16 concurrently via
`--territory-bytes 10000000` and a smaller `--workers` value.

## Next gate

The body has saturated, but validation throughput fell as active event rows
grew. The next optimization should preserve all developmental territories and
reduce candidate scanning inside a selected event row, rather than compressing
the global lookup structure. Any change must retain the language score and all
four non-language gates.

The durable run is stored under
`runs/alm_phase39_awake_competition_300m/` (ignored by Git). Twenty-seven
focused tests pass.
