"""Desenho de esqueleto e HUD — só para o operador conferir a captura.

Deliberadamente sem informação de classificação (sem prob_fall, sem alerta):
este app só captura, não roda o GRU.
"""

from __future__ import annotations

import cv2
import numpy as np

from capture_app.recording import Detection

# Pares de índices COCO-17 conectados por um "osso" do esqueleto.
_SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # ombros + braços
    (5, 11), (6, 12), (11, 12),                    # tronco
    (11, 13), (13, 15), (12, 14), (14, 16),        # pernas
    (0, 1), (0, 2), (1, 3), (2, 4),                # rosto
]

_JOINT_COLOR = (0, 255, 0)
_BONE_COLOR = (255, 200, 0)
_TEXT_COLOR = (255, 255, 255)
_REC_COLOR = (0, 0, 255)


def draw_skeleton(frame: np.ndarray, detections: list[Detection]) -> None:
    """Desenha esqueleto + id de cada pessoa detectada, in-place."""
    for det in detections:
        kp = det.keypoints
        for a, b in _SKELETON_EDGES:
            if kp[a, 2] > 0 and kp[b, 2] > 0:
                pt_a = (int(kp[a, 0]), int(kp[a, 1]))
                pt_b = (int(kp[b, 0]), int(kp[b, 1]))
                cv2.line(frame, pt_a, pt_b, _BONE_COLOR, 2)

        for x, y, conf in kp:
            if conf > 0:
                cv2.circle(frame, (int(x), int(y)), 3, _JOINT_COLOR, -1)

        valid_joints = kp[kp[:, 2] > 0]
        if len(valid_joints):
            top = valid_joints[np.argmin(valid_joints[:, 1])]
            cv2.putText(
                frame, f"id {det.person_id}", (int(top[0]), int(top[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _JOINT_COLOR, 1, cv2.LINE_AA,
            )


def draw_hud(
    frame: np.ndarray, *, scenario_label: str, scenario_id: str,
    take: int, n_frames: int, recording: bool,
) -> None:
    """HUD no topo: cenário, take, contagem de frames, indicador REC."""
    lines = [f"{scenario_id}  ({scenario_label})   take {take:02d}"]
    if recording:
        lines.append(f"REC  {n_frames} frames   [SPACE] salvar   [ESC] descartar")
    else:
        lines.append("pronto para gravar")

    for i, text in enumerate(lines):
        y = 25 + i * 22
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    _TEXT_COLOR, 1, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (frame.shape[1] - 20, 20), 8, _REC_COLOR, -1)
