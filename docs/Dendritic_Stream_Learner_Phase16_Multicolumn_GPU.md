# Phase 16: Multi-Column GPU Anatomy

## Outcome

The frozen GPU oracle now extends from one small column to the literal ALM-1GB
geometry. `CudaMultiColumnFrozenEvaluator` assigns one thread to each branch
within a causal depth stratum. A cooperative grid barrier acts like a laminar
boundary: lower-depth assemblies finish publishing before higher-depth
dendrites may read them. Each column conducts its own stable local tournament,
then a sparse soma phase selects active columns and the final relation vote.

This preserves the CPU ontology:

1. local dendrites respond;
2. branches compete inside a cortical column;
3. a small set of columns reaches the soma;
4. contextual residuals alter the successor distribution.

## Learned oracle

A two-column, 512-slot contextual organism was grown for 30,000 TinyStories
bytes. It developed 167 live relations. On a reset 4,096-byte prefix, the
multi-column CUDA kernel matched compiled CPU inference with maximum absolute
probability error `7.99e-8` and mean error `9.14e-9`.

The same test at 100,000 developmental bytes produced 460 live relations and
maximum error `1.12e-7`. Competition therefore agrees after both columns are
substantially occupied, not only in an empty-anatomy case.

## ALM-1GB geometry gate

An ALM class is its maximum serialized organism footprint, not a parameter
count. With the contextual-residual configuration:

- requested class: 1,000,000,000 bytes;
- byte-accounted maximum: 999,999,936 bytes;
- relation capacity: 112,341;
- cortical columns: 439;
- cooperative resident blocks: 168;
- active FP32 frozen-evaluator allocation: 244,967,424 bytes.

The RTX 5080 successfully loaded this geometry and advanced a causal
transition. The grid cannot keep 439 blocks resident simultaneously, so its
168 resident blocks stride over columns while retaining global depth barriers.
This removes a hidden launch ceiling that would otherwise have appeared
between ALM-256MB and ALM-1GB.

The active frozen mirror is smaller than the canonical body because it omits
nursery and local-plasticity storage. Total process VRAM was roughly 1.97GB,
including the CUDA context and allocator reservations. Neither number changes
the ALM's class: the canonical class remains the serialized biological body.
BF16 and scaled FP8 mirrors can later reduce VRAM further.

## Next gate

Frozen execution is no longer the scaling blocker. The next slice adds local
outcome learning to FP32 device masters. Developmental changes—birth, pruning,
consolidation, dormancy, and checkpointing—remain CPU barriers until GPU and
CPU make identical decisions on a fixed oracle stream. Only then should a
longer ALM-16MB or ALM-64MB TinyStories run move onto this backend.
