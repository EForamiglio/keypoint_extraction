"""Configuração centralizada: paths e defaults do app.

Mantém os módulos de lógica (scenario_catalog, recording, manifest, ...) livres
de conhecimento sobre onde as coisas ficam no disco ou quais são os defaults de
CLI — eles recebem um AppConfig já resolvido.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SCENARIOS_PATH = PROJECT_ROOT / "scenarios.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# Identifica a origem do dado no manifest.csv, para quem for ingerir isso num
# pipeline de treino mais amplo.
SOURCE_NAME = "keypoint_extraction"

NUM_KEYPOINTS = 17   # esqueleto COCO-17, mesma convenção do YOLO pose

# Abaixo deste valor de confiança, o keypoint individual é zerado (x=y=conf=0)
# em vez de descartado — mantém o vetor de tamanho fixo por frame.
DEFAULT_CONF_THRESHOLD = 0.25

# Pessoas rastreadas com menos frames válidos que isso na tomada são
# descartadas (provável track fantasma, não uma pessoa em cena por tempo
# suficiente para virar dado de treino útil).
DEFAULT_MIN_FRAMES_PER_PERSON = 10

# Tempo entre escolher o cenário e a gravação começar, para dar tempo de se
# posicionar antes do REC.
DEFAULT_COUNTDOWN_SECONDS = 3

# Usado só se a câmera não reportar um FPS válido (algumas webcams retornam 0).
FALLBACK_FPS = 30.0


@dataclass(frozen=True)
class AppConfig:
    """Configuração resolvida de uma execução do app."""

    camera_index: int = 0
    yolo_model: str = "yolov8n-pose.pt"
    yolo_conf: float = 0.25
    imgsz: int = 640
    device: str = "cpu"
    scenarios_path: Path = DEFAULT_SCENARIOS_PATH
    data_dir: Path = DEFAULT_DATA_DIR
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    min_frames_per_person: int = DEFAULT_MIN_FRAMES_PER_PERSON
    countdown_seconds: int = DEFAULT_COUNTDOWN_SECONDS

    @property
    def captures_dir(self) -> Path:
        return self.data_dir / "captures"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.csv"
