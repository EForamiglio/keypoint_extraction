from __future__ import annotations

import numpy as np
import pytest

from capture_app.config import AppConfig
from capture_app.scenario_catalog import Scenario


@pytest.fixture
def adl_scenario() -> Scenario:
    return Scenario(id="sentando", label="ADL", description="Sentar em uma cadeira")


@pytest.fixture
def fall_scenario() -> Scenario:
    return Scenario(id="queda_lado_esquerdo", label="FALL")


@pytest.fixture
def app_config(tmp_path) -> AppConfig:
    """AppConfig apontando para diretórios temporários e min_frames_per_person
    baixo, para não exigir tomadas longas nos testes."""
    return AppConfig(data_dir=tmp_path / "data", min_frames_per_person=2)


def _make_keypoints(seed: int = 0) -> np.ndarray:
    """(17,3) determinístico, só para os testes distinguirem uma pessoa da outra."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 480, size=(17, 3)).astype(np.float32)


@pytest.fixture
def make_keypoints():
    """Factory fixture: `make_keypoints(seed=1)` -> array (17,3) determinístico."""
    return _make_keypoints
