"""Entrypoint da aplicação de captura de cenários.

Uso básico:
    python main.py

Ver README.md para a lista completa de flags e o fluxo de uso.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from capture_app.cli_menu import run
from capture_app.config import (
    AppConfig,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_COUNTDOWN_SECONDS,
    DEFAULT_DATA_DIR,
    DEFAULT_MIN_FRAMES_PER_PERSON,
    DEFAULT_SCENARIOS_PATH,
)


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Captura keypoints de pose por cenário, para dataset de "
                    "treino/validação do classificador GRU de quedas."
    )
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Índice da câmera (padrão: 0)")
    parser.add_argument("--model", default="yolov8n-pose.pt",
                        help="Modelo YOLO pose (baixa automaticamente se necessário)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confiança mínima de detecção do YOLO")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="Limiar de IoU do NMS (menor = suprime mais caixas duplicadas)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Resolução de entrada do YOLO")
    parser.add_argument("--device", default="cpu",
                        help="Device de inferência: 'cpu' ou 'cuda'")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Diretório de saída (captures/ e manifest.csv)")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS_PATH,
                        help="Caminho do catálogo de cenários (scenarios.yaml)")
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD,
                        help="Abaixo disso, um keypoint individual é zerado")
    parser.add_argument("--min-frames-per-person", type=int,
                        default=DEFAULT_MIN_FRAMES_PER_PERSON,
                        help="Pessoas com menos frames válidos que isso são descartadas")
    parser.add_argument("--countdown", type=int, default=DEFAULT_COUNTDOWN_SECONDS,
                        help="Segundos de contagem regressiva antes de gravar")
    args = parser.parse_args()

    return AppConfig(
        camera_index=args.camera_index,
        yolo_model=args.model,
        yolo_conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        scenarios_path=args.scenarios,
        data_dir=args.data_dir,
        conf_threshold=args.conf_threshold,
        min_frames_per_person=args.min_frames_per_person,
        countdown_seconds=args.countdown,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
