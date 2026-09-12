"""Wrapper fino sobre cv2.VideoCapture.

Sem backend específico de plataforma (nada de cv2.CAP_DSHOW) — usa o backend
default do OpenCV, que funciona tanto em Windows quanto em macOS.
"""

from __future__ import annotations

import cv2
import numpy as np

from capture_app.config import FALLBACK_FPS


class CameraError(RuntimeError):
    """A câmera não pôde ser aberta ou parou de responder."""


class Camera:
    """Context manager: `with Camera(0) as cam: ...` libera o dispositivo ao sair."""

    def __init__(self, index: int) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise CameraError(
                f"Não foi possível abrir a câmera de índice {index}. "
                "Verifique se ela está conectada e se nenhum outro programa "
                "está usando (Zoom, Teams, outro processo deste app, etc)."
            )

    @property
    def fps(self) -> float:
        """FPS reportado pelo driver; algumas webcams retornam 0 ou valores
        inválidos, então cai para FALLBACK_FPS nesse caso."""
        reported = self._cap.get(cv2.CAP_PROP_FPS)
        return reported if reported and reported > 1.0 else FALLBACK_FPS

    def read(self) -> np.ndarray:
        """Lê um frame BGR. Levanta CameraError se a leitura falhar."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("Falha ao ler frame da câmera — dispositivo desconectado?")
        return frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
