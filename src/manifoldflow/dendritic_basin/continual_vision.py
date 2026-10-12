"""Bounded, local-learning vision adapter for chronological data streams.

The adapter deliberately keeps the benchmark contract separate from the
learner: images arrive as raw sensory vectors, predictions are made before a
label is revealed, and the revealed label can update only the currently awake
column.  No gradient tape, replay store, task identifier, or pretrained image
encoder is present.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import Tensor


@dataclass(frozen=True)
class ContinualVisionConfig:
    classes: int
    image_size: int = 24
    columns: int = 4
    context_decay: float = 0.94
    sleep_interval: int = 4096
    max_body_bytes: int = 64_000_000
    learning_rate: float = 0.5
    maximum_synapses: int = 128
    maximum_competitors: int = 16
    maximum_visual_assemblies: int = 50_000
    visual_birth_activation: float = 0.72
    visual_routing_width: int = 16
    relation_sketch_words: int = 4
    relation_turnover_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.classes < 2:
            raise ValueError("classes must be at least two")
        if self.image_size < 4 or self.columns < 1:
            raise ValueError("invalid receptor or column geometry")
        if not 0.0 <= self.context_decay < 1.0:
            raise ValueError("context_decay must be in [0, 1)")
        if self.sleep_interval < 1 or self.max_body_bytes < 1:
            raise ValueError("sleep interval and body budget must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.maximum_synapses < 1 or self.maximum_competitors < 0:
            raise ValueError("invalid local assembly capacity")
        if self.maximum_visual_assemblies < 1 or self.visual_routing_width < 1:
            raise ValueError("invalid visual assembly capacity")
        if not 0.0 < self.visual_birth_activation <= 1.0:
            raise ValueError("visual birth activation must be in (0, 1]")
        if self.relation_sketch_words < 1:
            raise ValueError("relation sketch must contain at least one word")
        if not 0.0 <= self.relation_turnover_fraction < 1.0:
            raise ValueError("relation turnover fraction must be in [0, 1)")


class RawImageReceptor:
    """Convert pixels to a fixed generic basal vector.

    This is receptor geometry, not a learned encoder.  The vector contains a
    resized RGB retinal sheet and its immediate horizontal and vertical local
    contrasts.  All values derive directly from the current image.
    """

    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)
        self.dimensions = 5 * self.image_size * self.image_size

    def __call__(self, image: Image.Image) -> np.ndarray:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        pixels = np.asarray(
            image.convert("RGB").resize(
                (self.image_size, self.image_size), resampling
            ),
            dtype=np.float32,
        ) / 255.0
        grey = pixels.mean(axis=2)
        horizontal = np.zeros_like(grey)
        vertical = np.zeros_like(grey)
        horizontal[:, 1:] = grey[:, 1:] - grey[:, :-1]
        vertical[1:, :] = grey[1:, :] - grey[:-1, :]
        vector = np.concatenate(
            (pixels.reshape(-1), horizontal.reshape(-1), vertical.reshape(-1))
        ).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector


class _LocalAssemblyColumn:
    """One locally updated cortical territory."""

    def __init__(
        self, config: ContinualVisionConfig, dimensions: int, device: torch.device
    ) -> None:
        self.config = config
        self.assemblies = torch.zeros(
            (config.classes, dimensions), dtype=torch.float32, device=device
        )
        self.counts = torch.zeros(config.classes, dtype=torch.int64, device=device)

    def scores(self, evidence: Tensor) -> Tensor:
        norms = torch.linalg.vector_norm(self.assemblies, dim=1)
        result = self.assemblies @ evidence
        return torch.where(norms > 0.0, torch.clamp(result, 0.0, 1.0), -torch.inf)

    def _update_row(self, evidence: Tensor, assembly: int, target: float) -> None:
        row = self.assemblies[assembly]
        active_count = min(self.config.maximum_synapses, evidence.numel())
        selected = torch.topk(torch.abs(evidence), active_count).indices
        prediction = torch.clamp(torch.dot(row, evidence), 0.0, 1.0)
        error = float(target) - prediction
        energy = torch.sum(evidence[selected] ** 2).clamp_min(1e-12)
        row[selected] += (
            self.config.learning_rate * error * evidence[selected] / energy
        )
        row.clamp_(-1.0, 1.0)
        live = torch.count_nonzero(row)
        if int(live.item()) > self.config.maximum_synapses:
            retained = torch.topk(torch.abs(row), self.config.maximum_synapses).indices
            mask = torch.zeros_like(row, dtype=torch.bool)
            mask[retained] = True
            row[~mask] = 0.0

    def observe_external(self, evidence: Tensor, label: int) -> None:
        """Apply a label-grounded local correction to one assembly."""
        scores = self.scores(evidence)
        competitors = torch.argsort(scores, descending=True)
        competitors = competitors[competitors != label][
            : self.config.maximum_competitors
        ]
        self._update_row(evidence, label, 1.0)
        for competitor in competitors:
            if torch.isfinite(scores[competitor]):
                self._update_row(evidence, int(competitor.item()), 0.0)
        self.counts[label] += 1


class ALM64MB4CContinualVision:
    """Latest-phase four-column, wake/sleep continual-vision phenotype.

    The class name denotes a hard serialized-body ceiling, not a promise that
    every run will occupy all 64 MB.  A slow internal trace supplies apical
    context to the same class assemblies that receive current basal evidence.
    Only an externally revealed class can alter anatomy.
    """

    def __init__(
        self, config: ContinualVisionConfig, *, device: str | torch.device = "cpu"
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.receptor = RawImageReceptor(config.image_size)
        self.cortical_dimensions = 2 * self.receptor.dimensions
        self.columns = [
            _LocalAssemblyColumn(
                config, self.cortical_dimensions, self.device
            )
            for _ in range(config.columns)
        ]
        self._sleep_base_assemblies = torch.zeros_like(self.columns[0].assemblies)
        self._sleep_base_counts = torch.zeros_like(self.columns[0].counts)
        self.slow_context = torch.zeros(
            self.receptor.dimensions, dtype=torch.float32, device=self.device
        )
        self.steps = 0
        self.sleep_cycles = 0
        self.external_updates = 0
        self._check_budget()

    @property
    def embodied_bytes(self) -> int:
        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    @property
    def active_synapses(self) -> int:
        return sum(
            int(torch.count_nonzero(column.assemblies).item())
            for column in self.columns
        )

    def _check_budget(self) -> None:
        size = self.embodied_bytes
        if size > self.config.max_body_bytes:
            raise RuntimeError(
                f"embodied size {size:,} exceeds {self.config.max_body_bytes:,}"
            )

    def reset_episode(self) -> None:
        self.slow_context.fill_(0.0)

    def evidence(self, image: Image.Image) -> Tensor:
        basal = torch.as_tensor(
            self.receptor(image), dtype=torch.float32, device=self.device
        )
        evidence = torch.cat((basal, self.slow_context))
        evidence = evidence / torch.clamp_min(torch.linalg.vector_norm(evidence), 1e-12)
        self.slow_context *= self.config.context_decay
        self.slow_context += (1.0 - self.config.context_decay) * basal
        return evidence

    def predict_evidence(self, evidence: Tensor) -> tuple[int, Tensor]:
        score_stack = torch.stack(
            [column.scores(evidence) for column in self.columns]
        )
        finite = torch.isfinite(score_stack)
        totals = torch.where(finite, score_stack, 0.0).sum(dim=0)
        support = finite.sum(dim=0)
        scores = torch.where(
            support > 0, totals / torch.clamp_min(support, 1), -torch.inf
        )
        prediction = int(torch.argmax(scores).item()) if torch.any(support) else 0
        return prediction, scores

    def predict(self, image: Image.Image) -> tuple[int, Tensor, Tensor]:
        evidence = self.evidence(image)
        prediction, scores = self.predict_evidence(evidence)
        return prediction, scores, evidence

    def observe_external(self, evidence: Tensor, label: int) -> None:
        if not 0 <= label < self.config.classes:
            raise ValueError("label outside configured classes")
        awake_column = self.steps % len(self.columns)
        self.columns[awake_column].observe_external(evidence, label)
        self.steps += 1
        self.external_updates += 1
        if self.steps % self.config.sleep_interval == 0:
            self.sleep()

    def sleep(self) -> None:
        """Consolidate independently acquired evidence, then clone phenotype."""
        deltas = [column.counts - self._sleep_base_counts for column in self.columns]
        total_counts = self._sleep_base_counts + torch.stack(deltas).sum(dim=0)
        merged_numerator = (
            self._sleep_base_assemblies * self._sleep_base_counts[:, None]
        )
        for column in self.columns:
            merged_numerator += (
                column.assemblies * column.counts[:, None]
                - self._sleep_base_assemblies * self._sleep_base_counts[:, None]
            )
        merged = torch.where(
            total_counts[:, None] > 0,
            merged_numerator / torch.clamp_min(total_counts[:, None], 1),
            0.0,
        )
        for row in merged:
            if int(torch.count_nonzero(row).item()) > self.config.maximum_synapses:
                retained = torch.topk(
                    torch.abs(row), self.config.maximum_synapses
                ).indices
                mask = torch.zeros_like(row, dtype=torch.bool)
                mask[retained] = True
                row[~mask] = 0.0
        for column in self.columns:
            column.assemblies[:] = merged
            column.counts[:] = total_counts
        self._sleep_base_assemblies[:] = merged
        self._sleep_base_counts[:] = total_counts
        self.sleep_cycles += 1
        self._check_budget()


class _GrowingVisualColumn:
    """Sparse sensory assemblies born from novelty and external mismatch."""

    def __init__(
        self, config: ContinualVisionConfig, dimensions: int, device: torch.device
    ) -> None:
        self.config = config
        self.dimensions = dimensions
        capacity = config.maximum_visual_assemblies
        synapses = config.maximum_synapses
        self.indices = torch.zeros(
            (capacity, synapses), dtype=torch.int32, device=device
        )
        self.weights = torch.zeros(
            (capacity, synapses), dtype=torch.float32, device=device
        )
        self.label_evidence = torch.zeros(
            (capacity, config.classes), dtype=torch.float32, device=device
        )
        self.observations = torch.zeros(
            capacity, dtype=torch.int32, device=device
        )
        self.relation_sketch = torch.zeros(
            (capacity, config.relation_sketch_words),
            dtype=torch.int64,
            device=device,
        )
        self.live = 0
        self.births = 0

    @property
    def embodied_bytes(self) -> int:
        payload = (
            self.indices[: self.live].detach().cpu().numpy(),
            self.weights[: self.live].detach().cpu().numpy(),
            self.label_evidence[: self.live].detach().cpu().numpy(),
            self.observations[: self.live].detach().cpu().numpy(),
            self.relation_sketch[: self.live].detach().cpu().numpy(),
        )
        return len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    def activations(self, evidence: Tensor) -> Tensor:
        if self.live == 0:
            return torch.empty(0, dtype=torch.float32, device=evidence.device)
        indexes = self.indices[: self.live].long()
        return torch.clamp(
            torch.sum(self.weights[: self.live] * evidence[indexes], dim=1),
            0.0,
            1.0,
        )

    def votes(self, evidence: Tensor) -> tuple[Tensor, Tensor]:
        activity = self.activations(evidence)
        result = torch.zeros(
            self.config.classes, dtype=torch.float32, device=evidence.device
        )
        if activity.numel() == 0:
            return result, activity
        width = min(self.config.visual_routing_width, len(activity))
        strengths, assemblies = torch.topk(activity, width)
        supported = strengths > 0.0
        strengths = strengths[supported]
        assemblies = assemblies[supported]
        if len(assemblies) == 0:
            return result, activity
        labels = self.label_evidence[assemblies]
        totals = labels.sum(dim=1, keepdim=True).clamp_min(1.0)
        result = torch.sum(strengths[:, None] * labels / totals, dim=0)
        return result, activity

    def _sensory_row(self, evidence: Tensor) -> tuple[Tensor, Tensor]:
        count = min(self.config.maximum_synapses, evidence.numel())
        indexes = torch.topk(torch.abs(evidence), count).indices
        values = evidence[indexes]
        energy = torch.sum(values**2).clamp_min(1e-12)
        return indexes.to(torch.int32), values / energy

    def _grow(self, evidence: Tensor, label: int) -> int | None:
        if self.live >= self.config.maximum_visual_assemblies:
            return None
        assembly = self.live
        indexes, weights = self._sensory_row(evidence)
        self.indices[assembly] = indexes
        self.weights[assembly] = weights
        self.label_evidence[assembly, label] = 1.0
        self.observations[assembly] = 1
        self.live += 1
        self.births += 1
        return assembly

    def _strengthen_relations(self, activity: Tensor, born: int | None) -> None:
        width = min(self.config.visual_routing_width, len(activity))
        if width:
            strengths, assemblies = torch.topk(activity, width)
            assemblies = assemblies[strengths > 0.0]
        else:
            assemblies = torch.empty(0, dtype=torch.int64, device=activity.device)
        if born is not None:
            assemblies = torch.cat(
                (
                    assemblies,
                    torch.tensor([born], dtype=torch.int64, device=activity.device),
                )
            )
        assemblies = torch.unique(assemblies)
        if len(assemblies) < 2:
            return
        signature = torch.zeros(
            self.config.relation_sketch_words,
            dtype=torch.int64,
            device=activity.device,
        )
        for assembly in assemblies.detach().cpu().tolist():
            word = int(assembly) % self.config.relation_sketch_words
            bit = (int(assembly) // self.config.relation_sketch_words) % 62
            signature[word] |= 1 << bit
        self.relation_sketch[assemblies] = torch.bitwise_or(
            self.relation_sketch[assemblies], signature
        )

    def observe_external(self, evidence: Tensor, label: int) -> bool:
        votes, activity = self.votes(evidence)
        prediction = int(torch.argmax(votes).item()) if torch.any(votes) else -1
        peak = float(torch.max(activity).item()) if len(activity) else 0.0
        if peak < self.config.visual_birth_activation or prediction != label:
            born = self._grow(evidence, label)
            self._strengthen_relations(activity, born)
            return born is not None

        winner = int(torch.argmax(activity).item())
        indexes = self.indices[winner].long()
        values = evidence[indexes]
        energy = torch.sum(values**2).clamp_min(1e-12)
        target = values / energy
        rate = self.config.learning_rate / (
            1.0 + float(self.observations[winner].item()) ** 0.5
        )
        self.weights[winner] += rate * (target - self.weights[winner])
        self.label_evidence[winner, label] += 1.0
        self.observations[winner] += 1
        self._strengthen_relations(activity, None)
        return False

    def load_rows(
        self,
        indices: Tensor,
        weights: Tensor,
        labels: Tensor,
        observations: Tensor,
        relations: Tensor,
    ) -> None:
        count = len(indices)
        if count > self.config.maximum_visual_assemblies:
            raise ValueError("merged visual tissue exceeds row capacity")
        self.live = count
        self.indices[:count] = indices
        self.weights[:count] = weights
        self.label_evidence[:count] = labels
        self.observations[:count] = observations
        self.relation_sketch[:count] = relations
        if count < self.config.maximum_visual_assemblies:
            self.weights[count:].zero_()
            self.label_evidence[count:].zero_()
            self.observations[count:].zero_()
            self.relation_sketch[count:].zero_()


class ALM64MB4CGrowingVision:
    """Four-lineage sparse visual organism with measured structural growth.

    Sensory assemblies are not class rows. They are born from the current raw
    basal/apical evidence, and a later external label develops a separate local
    association. Sleep merges the independently acquired rows, removes weak
    excess tissue if necessary, and clones one bounded mature phenotype.
    """

    def __init__(
        self, config: ContinualVisionConfig, *, device: str | torch.device = "cpu"
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.receptor = RawImageReceptor(config.image_size)
        self.cortical_dimensions = 2 * self.receptor.dimensions
        self.columns = [
            _GrowingVisualColumn(config, self.cortical_dimensions, self.device)
            for _ in range(config.columns)
        ]
        self.slow_context = torch.zeros(
            self.receptor.dimensions, dtype=torch.float32, device=self.device
        )
        self.label_counts = torch.zeros(
            config.classes, dtype=torch.float32, device=self.device
        )
        self.steps = 0
        self.sleep_cycles = 0
        self.external_updates = 0
        self.visual_births = 0
        self.relation_turnover_releases = 0
        self._base_rows = 0

    @property
    def embodied_bytes(self) -> int:
        # The mature phenotype is one consolidated column; 4C denotes wake
        # lineages, not four permanent inference copies.
        payload = (
            self.config,
            self.columns[0].indices[: self.columns[0].live].detach().cpu().numpy(),
            self.columns[0].weights[: self.columns[0].live].detach().cpu().numpy(),
            self.columns[0]
            .label_evidence[: self.columns[0].live]
            .detach()
            .cpu()
            .numpy(),
            self.columns[0]
            .observations[: self.columns[0].live]
            .detach()
            .cpu()
            .numpy(),
            self.columns[0]
            .relation_sketch[: self.columns[0].live]
            .detach()
            .cpu()
            .numpy(),
        )
        return len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    @property
    def live_assemblies(self) -> int:
        return max(column.live for column in self.columns)

    @property
    def active_synapses(self) -> int:
        return self.live_assemblies * self.config.maximum_synapses

    def reset_episode(self) -> None:
        self.slow_context.zero_()

    def evidence(self, image: Image.Image) -> Tensor:
        basal = torch.as_tensor(
            self.receptor(image), dtype=torch.float32, device=self.device
        )
        evidence = torch.cat((basal, self.slow_context))
        evidence /= torch.linalg.vector_norm(evidence).clamp_min(1e-12)
        self.slow_context *= self.config.context_decay
        self.slow_context += (1.0 - self.config.context_decay) * basal
        return evidence

    def predict_evidence(self, evidence: Tensor) -> tuple[int, Tensor]:
        votes = torch.zeros(
            self.config.classes, dtype=torch.float32, device=self.device
        )
        for column in self.columns:
            local, _activity = column.votes(evidence)
            votes += local
        if not torch.any(votes):
            votes = self.label_counts.clone()
        prediction = int(torch.argmax(votes).item())
        return prediction, votes

    def predict(self, image: Image.Image) -> tuple[int, Tensor, Tensor]:
        evidence = self.evidence(image)
        prediction, scores = self.predict_evidence(evidence)
        return prediction, scores, evidence

    def observe_external(self, evidence: Tensor, label: int) -> None:
        if not 0 <= label < self.config.classes:
            raise ValueError("label outside configured classes")
        awake = self.columns[self.steps % len(self.columns)]
        born = awake.observe_external(evidence, label)
        self.visual_births += int(born)
        self.label_counts[label] += 1.0
        self.steps += 1
        self.external_updates += 1
        if self.steps % self.config.sleep_interval == 0:
            self.sleep()

    def _merged_rows(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        base = self._base_rows
        if base:
            base_indices = self.columns[0].indices[:base].clone()
            base_weights = torch.stack(
                [column.weights[:base] for column in self.columns]
            ).mean(dim=0)
            base_labels = torch.stack(
                [column.label_evidence[:base] for column in self.columns]
            ).mean(dim=0)
            base_observations = torch.stack(
                [column.observations[:base].float() for column in self.columns]
            ).mean(dim=0).to(torch.int32)
            base_relations = self.columns[0].relation_sketch[:base].clone()
            for column in self.columns[1:]:
                base_relations = torch.bitwise_or(
                    base_relations, column.relation_sketch[:base]
                )
        else:
            base_indices = self.columns[0].indices[:0]
            base_weights = self.columns[0].weights[:0]
            base_labels = self.columns[0].label_evidence[:0]
            base_observations = self.columns[0].observations[:0]
            base_relations = self.columns[0].relation_sketch[:0]
        def merged(base_part: Tensor, name: str) -> Tensor:
            additions = tuple(
                getattr(column, name)[base : column.live]
                for column in self.columns
            )
            return torch.cat((base_part, *additions), dim=0)

        return tuple(
            merged(base_part, name)
            for base_part, name in (
                (base_indices, "indices"),
                (base_weights, "weights"),
                (base_labels, "label_evidence"),
                (base_observations, "observations"),
                (base_relations, "relation_sketch"),
            )
        )  # type: ignore[return-value]

    def sleep(self) -> None:
        indices, weights, labels, observations, relations = self._merged_rows()
        capacity = self.config.maximum_visual_assemblies
        release = (
            int(capacity * self.config.relation_turnover_fraction)
            if len(indices) >= capacity
            else 0
        )
        retain_count = min(len(indices), capacity - release)
        if len(indices) > retain_count:
            relation_bytes = relations.detach().cpu().numpy().view(np.uint8)
            relation_degree = np.unpackbits(relation_bytes, axis=-1).sum(axis=1)
            degree = torch.as_tensor(
                relation_degree, dtype=torch.float32, device=self.device
            )
            confidence = labels.max(dim=1).values / labels.sum(dim=1).clamp_min(1.0)
            utility = degree * (capacity + 1.0) + observations.float() * confidence
            retained = torch.topk(utility, retain_count).indices
            indices = indices[retained]
            weights = weights[retained]
            labels = labels[retained]
            observations = observations[retained]
            relations = relations[retained]
            self.relation_turnover_releases += len(utility) - retain_count
        for column in self.columns:
            column.load_rows(indices, weights, labels, observations, relations)
        self._base_rows = len(indices)
        self.sleep_cycles += 1
        if self.embodied_bytes > self.config.max_body_bytes:
            raise RuntimeError("consolidated visual phenotype exceeds body class")
