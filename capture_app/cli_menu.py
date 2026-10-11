"""Loop de UX: escolher cenário -> contagem regressiva -> gravar -> salvar -> repetir.

Câmera e modelo YOLO são inicializados uma única vez fora do loop — reabri-los
a cada tomada adicionaria uma fricção que vai contra o propósito do app
(iteração rápida de testes e retestes).
"""

from __future__ import annotations

import math
import time

import cv2

from capture_app.camera import Camera, CameraError
from capture_app.config import AppConfig
from capture_app.manifest import ManifestEntry, append_entries, takes_for_scenario
from capture_app.overlay import draw_hud, draw_skeleton
from capture_app.pose_source import PoseSource
from capture_app.recording import RecordingSession
from capture_app.scenario_catalog import Scenario, load_catalog
from capture_app.session_ids import new_session_id
from capture_app.take_numbering import next_take_number, takes_from_filenames

_WINDOW_NAME = "keypoint_extraction"
_SPACE, _ESC = ord(" "), 27


def run(config: AppConfig) -> None:
    scenarios = load_catalog(config.scenarios_path)
    session_id = new_session_id()
    pose_source = PoseSource(
        config.yolo_model, config.yolo_conf, config.imgsz, config.device,
        config.conf_threshold, iou=config.iou,
    )

    last_scenario_id: str | None = None

    try:
        while True:
            scenario = _prompt_menu(scenarios, config, last_scenario_id)
            if scenario is None:
                break
            last_scenario_id = scenario.id
            take = _next_take(config, scenario.id)

            # A câmera só fica aberta durante a tomada. Segurá-la aberta e
            # parada no menu fazia o Windows invalidar o dispositivo após
            # alguns minutos ocioso (MSMF -1072873822), e o app seguia com um
            # objeto de câmera morto, falhando em toda tomada seguinte.
            print("Abrindo câmera...")
            try:
                with Camera(config.camera_index) as camera:
                    _countdown(config.countdown_seconds, scenario, camera)
                    session = _record_take(camera, pose_source, scenario, take, config)
                    fps = camera.fps
            except CameraError as exc:
                print(f"\nErro de câmera: {exc}\nNada foi gravado; volte ao menu e tente de novo.")
                continue

            if session is not None:
                _save_and_report(session, scenario, take, session_id, fps, config)
    finally:
        cv2.destroyAllWindows()


def _prompt_menu(
    scenarios: list[Scenario], config: AppConfig, last_scenario_id: str | None,
) -> Scenario | None:
    print()
    print("Cenários disponíveis:")
    for i, scenario in enumerate(scenarios, start=1):
        take = _next_take(config, scenario.id)
        print(f"  [{i}] {scenario.id:<28} ({scenario.label})   próximo take: {take:02d}")
    if last_scenario_id:
        print("  [r] repetir último cenário usado")
    print("  [q] sair")

    while True:
        raw = input("> ").strip().lower()
        if raw == "q":
            return None
        if raw == "r" and last_scenario_id:
            return next(s for s in scenarios if s.id == last_scenario_id)
        if raw.isdigit() and 1 <= int(raw) <= len(scenarios):
            return scenarios[int(raw) - 1]
        print("Opção inválida.")


def _next_take(config: AppConfig, scenario_id: str) -> int:
    from_manifest = takes_for_scenario(config.manifest_path, scenario_id)
    from_disk = takes_from_filenames(config.captures_dir / scenario_id, scenario_id)
    return next_take_number(from_manifest | from_disk)


def _countdown(seconds: int, scenario: Scenario, camera: Camera) -> None:
    """Contagem regressiva que lê (e descarta) frames da câmera: estabiliza a
    exposição automática e evita que frames velhos do buffer entrem no início
    da tomada."""
    deadline = time.monotonic() + seconds
    last_shown = None
    while (remaining := deadline - time.monotonic()) > 0:
        camera.read()
        shown = math.ceil(remaining)
        if shown != last_shown:
            print(f"Gravando '{scenario.id}' em {shown}...", end="\r")
            last_shown = shown
    print(" " * 60, end="\r")


def _record_take(
    camera: Camera, pose_source: PoseSource, scenario: Scenario, take: int,
    config: AppConfig,
) -> RecordingSession | None:
    """Roda a janela de gravação. Retorna None se abortado (ESC) ou se a
    câmera falhar no meio da tomada — nesses casos nada é salvo."""
    session = RecordingSession(scenario, take, config)
    cv2.namedWindow(_WINDOW_NAME)
    try:
        while True:
            try:
                frame = camera.read()
            except CameraError as exc:
                print(f"\nErro de câmera durante a gravação: {exc}")
                return None

            detections = pose_source.detect(frame)
            session.add_frame(detections, timestamp=time.perf_counter())

            draw_skeleton(frame, detections)
            draw_hud(
                frame, scenario_label=scenario.label, scenario_id=scenario.id,
                take=take, n_frames=session.n_frames, recording=True,
            )
            cv2.imshow(_WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (_SPACE, ord("s")):
                return session
            if key == _ESC:
                print("\nTomada descartada.")
                return None
    finally:
        cv2.destroyWindow(_WINDOW_NAME)


def _save_and_report(
    session: RecordingSession, scenario: Scenario, take: int, session_id: str,
    fps: float, config: AppConfig,
) -> None:
    out_dir = config.captures_dir / scenario.id
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = session.finalize(out_dir)

    if not clips:
        print("Nenhuma pessoa com frames suficientes na tomada — nada foi salvo.")
        return

    group_id = f"{scenario.id}_take{take:02d}"
    entries = [
        ManifestEntry(
            clip_id=clip.clip_id,
            is_fall=scenario.is_fall,
            # fps medido na tomada; o do driver só entra se não houve timestamps
            src_fps=round(clip.measured_fps, 2) if clip.measured_fps else fps,
            lbl_path=clip.labels_path.resolve(),
            group_id=group_id,
            px_path=clip.keypoints_path.resolve(),
            scenario_id=scenario.id,
            scenario_label=scenario.label,
            take_number=take,
            person_idx=clip.person_idx,
            n_frames=clip.n_frames,
            session_id=session_id,
        )
        for clip in clips
    ]
    append_entries(config.manifest_path, entries)

    measured = clips[0].measured_fps
    duration = f"{clips[0].n_frames / measured:.1f}s a {measured:.1f} fps" if measured else "sem timestamps"
    print(f"Salvo: {len(clips)} clipe(s) -> {[c.clip_id for c in clips]}  ({duration})")
