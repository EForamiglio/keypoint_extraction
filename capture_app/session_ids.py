"""Geração de identificadores de sessão.

Isolado do resto do app para ser testável com um relógio determinístico —
`main.py`/`cli_menu.py` chamam `new_session_id()` sem argumento; os testes
passam `now` explicitamente.
"""

from __future__ import annotations

from datetime import datetime


def new_session_id(now: datetime | None = None) -> str:
    """Ex.: '20260912_143205'. Uma sessão = uma execução do app."""
    now = now or datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")
