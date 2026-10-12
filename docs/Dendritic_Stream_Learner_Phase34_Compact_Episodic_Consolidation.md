# Phase 34: Compact Episodic Consolidation

## Incremental anatomy merge

The first episodic merger rebuilt every previously accumulated branch whenever
another cortical column arrived. With 30 columns this made consolidation
quadratic in accumulated living anatomy and retained all completed column
objects until training ended.

The compact merger now remaps and mutates only paths present in the incoming
column. Living branches join living branches in place, incoming living tissue
promotes an aligned nursery path directly, and nursery evidence touches only
its matching candidate. Completed worker anatomy is consolidated as soon as
the process finishes and then released, reducing peak parent-process memory.

## Exact benchmark

On the repeated 30 x 1MB episodic experiment:

| Measurement | Original merge | Compact merge |
|---|---:|---:|
| Held-out loss | 2.313817 bpb | 2.313817 bpb |
| Final body | 3,030,625 B | 3,030,625 B |
| Episodic branches | 39,423 | 39,423 |
| Episodic synapses | 120,053 | 120,053 |
| Globally pruned branches | 110,687 | 110,687 |
| Consolidation time | 15.030 s | 3.387 s |

The compact path is 4.44x faster with identical learned anatomy and behavior.
A four-column streaming-merge check likewise reproduced every metric while
reducing consolidation from 287ms to 111ms.

## Rejected local pruning

Pruning non-positive branches independently inside each 1MB territory removed
649,413 local branches and shortened final merge time, but worsened held-out
loss from 2.313817 to 2.324298 bpb. Globally useful evidence can be weak inside
one territory, and removed branches regrow. The full run therefore retains
local evidence until cross-column consensus, then performs one utility-gated
metabolic consolidation.

## Full-run memory correction

The first full launch assigned one 64.1MB territory to each of 30 workers.
Although the final body would be compact, each worker retained its unpruned
Python object graph until territory completion. Partway through training the
workers owned approximately 32GB of private committed memory on a 30.6GB
machine, with only 6.9GB of virtual commit remaining. The attempt was stopped
before paging or allocation failure; it had not reached a checkpoint.

Body size and developmental working memory are therefore separate measures.
The body is serialized, globally consolidated anatomy. Developmental memory
also includes Python interpreter state, boxed integers, tuple keys, dictionary
capacity, nursery tissue, and 30 independent pre-consensus graphs.

The corrected runner supports bounded territories. For the full corpus, 963
territories of at most 2MB stream through persistent workers. Each completed
territory merges immediately into the global organ and its local graph is
released. A matched 30MB acceptance run using 15 x 2MB territories reached
2.310467 bpb versus 2.313817 for the earlier layout, with a 3,159,312-byte
post-pruning body. This bounds worker anatomy without the behavioral damage of
local pruning.

A second launch showed that 30 spawned Windows/Python runtimes themselves
reserve roughly 29GB of private commit before substantial anatomical growth,
leaving only 9.8GB of virtual commit for the global graph. The production run
therefore uses 16 persistent workers. After the first reuse cycle they occupied
15.24GB private, leaving 23.4GB virtual commit free, and merged the first 18MB
at 428KB/s. The lower worker count protects room for consolidated anatomy and
was faster initially because it reduced contention with the older oracle.
