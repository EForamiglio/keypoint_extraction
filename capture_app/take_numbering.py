"""Numeração de takes por cenário — lógica pura, sem I/O.

Reconcilia duas fontes de verdade: o manifest.csv (via `manifest.py`, que
fornece o `set[int]` de takes já registrados) e os nomes de arquivo já no
disco (caso o manifest tenha sido perdido/corrompido, ou uma gravação
interrompida entre salvar o `.npy` e escrever a linha do manifest). O próximo
take é sempre maior que o maior número visto em qualquer uma das duas fontes,
para nunca sobrescrever uma gravação existente.
"""

from __future__ import annotations

import re
from pathlib import Path

_TAKE_RE = re.compile(r"_take(\d+)_p\d+\.npy$")


def next_take_number(existing: set[int]) -> int:
    """Próximo número de take, começando em 1."""
    return max(existing, default=0) + 1


def takes_from_filenames(scenario_dir: Path, scenario_id: str) -> set[int]:
    """Números de take encontrados nos nomes de arquivo `.npy` de keypoints
    (não `_labels.npy`, para não contar cada take em dobro) de um diretório
    de cenário. Retorna vazio se o diretório ainda não existe.
    """
    if not scenario_dir.exists():
        return set()

    prefix = f"{scenario_id}_take"
    takes: set[int] = set()
    for path in scenario_dir.glob(f"{prefix}*.npy"):
        match = _TAKE_RE.search(path.name)
        if match:
            takes.add(int(match.group(1)))
    return takes
