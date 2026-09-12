"""Wrapper do YOLO pose com tracking (BotSort via Ultralytics `.track()`).

Padrão inspirado em `fall-classifier-bench/pose_detector.py` (projeto irmão),
copiado e não importado — este projeto é independente de propósito.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from capture_app.config import NUM_KEYPOINTS
from capture_app.recording import Detection


class PoseSource:
    """Detecta pose + person_id (tracking) por frame via Ultralytics YOLO.

    O tracking de identidade (`person_id`) vem embutido no `.track()` do
    Ultralytics (BotSort) — não é implementado manualmente aqui.
    """

    def __init__(
        self,
        model_path: str,
        conf: float,
        imgsz: int,
        device: str,
        conf_threshold: float,
    ) -> None:
        self._model = YOLO(model_path)
        self._conf = conf
        self._imgsz = imgsz
        self._device = device
        self._conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Roda pose + tracking num frame BGR.

        Retorna uma Detection por pessoa rastreada. Joints individuais abaixo
        de `conf_threshold` ficam zerados (x=y=conf=0) em vez de descartados,
        mantendo o vetor (17,3) de tamanho fixo por pessoa — mesma convenção
        usada no treino do classificador GRU deste projeto de PFC.
        """
        results = self._model.track(
            frame,
            conf=self._conf,
            imgsz=self._imgsz,
            device=self._device,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
        )
        result = results[0]
        if result.keypoints is None or result.boxes is None or result.boxes.id is None:
            return []

        person_ids = result.boxes.id.int().cpu().numpy()
        keypoints_xy = result.keypoints.xy.cpu().numpy()        # (N, 17, 2)

        conf_tensor = result.keypoints.conf
        if conf_tensor is None:
            keypoints_conf = np.ones(keypoints_xy.shape[:2], dtype=np.float32)
        else:
            keypoints_conf = conf_tensor.cpu().numpy()          # (N, 17)

        detections: list[Detection] = []
        for i, person_id in enumerate(person_ids):
            kp = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
            kp[:, :2] = keypoints_xy[i]
            kp[:, 2] = keypoints_conf[i]
            kp[kp[:, 2] < self._conf_threshold] = 0.0
            detections.append(Detection(person_id=int(person_id), keypoints=kp))

        return detections
