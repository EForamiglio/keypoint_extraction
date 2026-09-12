"""Monta os arrays de keypoints por pessoa a partir de uma tomada de captura.

Mantido livre de I/O de câmera e de YOLO: recebe listas de `Detection` (uma
lista por frame processado) e produz os arrays `(T,17,3)` + labels + metadados
que `manifest.py`/`cli_menu.py` gravam em disco. Isso é o que torna a lógica
mais sensível a bug silencioso (zero-fill de frames ausentes, geração de
clip_id, descarte de tracks curtos) testável sem hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from capture_app.config import AppConfig, NUM_KEYPOINTS
from capture_app.scenario_catalog import Scenario


@dataclass(frozen=True)
class Detection:
    """Uma pessoa detectada em um frame.

    keypoints: (17,3) x,y,conf em pixels, COCO-17. Joints abaixo do limiar de
    confiança já devem chegar zerados (responsabilidade de pose_source.py) —
    aqui só se monta o array por pessoa ao longo do tempo.
    """

    person_id: int
    keypoints: np.ndarray


@dataclass(frozen=True)
class ClipResult:
    """Metadados de um clipe pronto para virar linha do manifest."""

    clip_id: str
    person_idx: int
    n_frames: int
    keypoints_path: Path
    labels_path: Path


class RecordingSession:
    """Acumula frames de uma tomada e produz os clipes por pessoa ao finalizar."""

    def __init__(self, scenario: Scenario, take: int, config: AppConfig) -> None:
        self.scenario = scenario
        self.take = take
        self._config = config
        self._frames: list[list[Detection]] = []

    def add_frame(self, detections: list[Detection]) -> None:
        self._frames.append(detections)

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def finalize(self, out_dir: Path) -> list[ClipResult]:
        """Agrupa por person_id, zero-preenche ausências, filtra tracks curtos
        e grava `.npy`/`_labels.npy` em `out_dir` (já deve existir).

        Pessoas com menos de `min_frames_per_person` frames válidos na tomada
        são descartadas — provável track fantasma do tracker, não uma pessoa
        em cena por tempo suficiente para virar dado de treino útil.
        """
        by_person = self._group_by_person()
        label_value = 2 if self.scenario.is_fall else 0
        results: list[ClipResult] = []

        for person_idx, (_person_id, per_frame) in enumerate(sorted(by_person.items())):
            n_valid = sum(1 for kp in per_frame if kp is not None)
            if n_valid < self._config.min_frames_per_person:
                continue

            keypoints = np.zeros((self.n_frames, NUM_KEYPOINTS, 3), dtype=np.float32)
            for t, kp in enumerate(per_frame):
                if kp is not None:
                    keypoints[t] = kp
            labels = np.full(self.n_frames, label_value, dtype=np.int8)

            clip_id = f"{self.scenario.id}_take{self.take:02d}_p{person_idx}"
            keypoints_path = out_dir / f"{clip_id}.npy"
            labels_path = out_dir / f"{clip_id}_labels.npy"
            np.save(keypoints_path, keypoints)
            np.save(labels_path, labels)

            results.append(ClipResult(
                clip_id=clip_id,
                person_idx=person_idx,
                n_frames=self.n_frames,
                keypoints_path=keypoints_path,
                labels_path=labels_path,
            ))

        return results

    def _group_by_person(self) -> dict[int, list[np.ndarray | None]]:
        """person_id -> lista de tamanho n_frames, com None nos frames em que
        a pessoa não apareceu."""
        person_ids = {det.person_id for frame in self._frames for det in frame}
        by_person: dict[int, list[np.ndarray | None]] = {
            pid: [None] * self.n_frames for pid in person_ids
        }
        for t, frame in enumerate(self._frames):
            for det in frame:
                by_person[det.person_id][t] = det.keypoints
        return by_person
