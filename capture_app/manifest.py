"""Leitura/escrita do manifest.csv — um registro por clipe gravado.

CSV puro via stdlib (sem pandas): o volume de linhas é pequeno — uma sessão de
captura gera dezenas de tomadas, não milhares — e `csv.DictWriter`/`DictReader`
já cobrem o que é preciso.

As colunas seguem os nomes que `modelos/data_prep/build_windows_v2.py::load_meta()`
já espera de uma fonte de dados (`clip_id, source, is_fall, src_fps, lbl_path,
group_id, px_path`), mais colunas extras de uso interno deste app — leitura por
nome não é afetada por colunas a mais.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from capture_app.config import SOURCE_NAME

FIELDNAMES = [
    "clip_id", "source", "is_fall", "src_fps", "lbl_path", "group_id", "px_path",
    "scenario_id", "scenario_label", "take_number", "person_idx", "n_frames",
    "session_id", "recorded_at", "operator_note",
]


@dataclass(frozen=True)
class ManifestEntry:
    """Uma linha do manifest. Campos sem default primeiro (regra do dataclass)."""

    clip_id: str
    is_fall: bool
    src_fps: float
    lbl_path: Path
    group_id: str
    px_path: Path
    scenario_id: str
    scenario_label: str
    take_number: int
    person_idx: int
    n_frames: int
    session_id: str
    operator_note: str = ""
    source: str = SOURCE_NAME
    recorded_at: str = ""

    def __post_init__(self) -> None:
        if not self.recorded_at:
            # frozen=True impede atribuição normal; contornado de propósito
            # só para preencher o timestamp default no momento da criação.
            object.__setattr__(
                self, "recorded_at",
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )

    def to_row(self) -> dict[str, str]:
        """Serializa para o formato de texto que o CSV grava."""
        row = asdict(self)
        row["is_fall"] = str(self.is_fall)
        row["src_fps"] = str(self.src_fps)
        row["lbl_path"] = str(self.lbl_path)
        row["px_path"] = str(self.px_path)
        row["take_number"] = str(self.take_number)
        row["person_idx"] = str(self.person_idx)
        row["n_frames"] = str(self.n_frames)
        return row


def ensure_manifest(path: Path) -> None:
    """Cria o CSV com header se ainda não existir. Idempotente."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=FIELDNAMES).writeheader()


def append_entries(path: Path, entries: list[ManifestEntry]) -> None:
    """Acrescenta linhas. O fechamento do arquivo ao sair do `with` já garante
    o flush — importante porque uma sessão de captura roda por várias tomadas
    e uma interrupção (Ctrl+C) entre uma tomada e outra não pode perder as
    linhas já gravadas."""
    ensure_manifest(path)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        for entry in entries:
            writer.writerow(entry.to_row())


def read_entries(path: Path) -> list[dict[str, str]]:
    """Lê todas as linhas já gravadas. Lista vazia se o arquivo não existe."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def takes_for_scenario(path: Path, scenario_id: str) -> set[int]:
    """Números de take já usados para um cenário, a partir do manifest."""
    return {
        int(row["take_number"])
        for row in read_entries(path)
        if row["scenario_id"] == scenario_id
    }
