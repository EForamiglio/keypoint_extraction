"""Carrega e valida o catálogo de cenários (scenarios.yaml).

Validação acontece toda na carga, não no uso — mais fácil corrigir um
scenarios.yaml malformado antes de começar uma sessão de captura do que
descobrir o problema no meio dela.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")
_VALID_LABELS = {"ADL", "FALL"}


class ScenarioCatalogError(ValueError):
    """Erro de parsing ou validação de scenarios.yaml."""


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str          # "ADL" ou "FALL"
    description: str = ""

    @property
    def is_fall(self) -> bool:
        return self.label == "FALL"


def load_catalog(path: Path) -> list[Scenario]:
    """Lê e valida scenarios.yaml. Levanta ScenarioCatalogError se inválido."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("scenarios")
    if not entries:
        raise ScenarioCatalogError(f"Nenhum cenário encontrado em {path}")

    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        scenario = _parse_entry(entry, index)
        if scenario.id in seen_ids:
            raise ScenarioCatalogError(f"id duplicado no catálogo: '{scenario.id}'")
        seen_ids.add(scenario.id)
        scenarios.append(scenario)

    return scenarios


def by_id(scenarios: list[Scenario]) -> dict[str, Scenario]:
    """Índice id -> Scenario, para lookups no menu e na gravação."""
    return {s.id: s for s in scenarios}


def _parse_entry(entry: dict, index: int) -> Scenario:
    if not isinstance(entry, dict):
        raise ScenarioCatalogError(f"entrada #{index}: formato inválido")

    scenario_id = entry.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ScenarioCatalogError(f"entrada #{index}: 'id' ausente ou inválido")
    if not _SLUG_RE.match(scenario_id):
        raise ScenarioCatalogError(
            f"id '{scenario_id}': deve conter só minúsculas, números e '_' "
            f"(regex {_SLUG_RE.pattern})"
        )

    label = entry.get("label")
    if label not in _VALID_LABELS:
        raise ScenarioCatalogError(
            f"cenário '{scenario_id}': label '{label}' inválido — "
            f"deve ser um de {sorted(_VALID_LABELS)}"
        )

    description = entry.get("description") or ""
    return Scenario(id=scenario_id, label=label, description=description)
