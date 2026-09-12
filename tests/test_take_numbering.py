from pathlib import Path

import numpy as np

from capture_app.take_numbering import next_take_number, takes_from_filenames


def test_next_take_number_starts_at_one_when_empty() -> None:
    assert next_take_number(set()) == 1


def test_next_take_number_is_one_past_the_highest_seen() -> None:
    assert next_take_number({1, 2, 5}) == 6


def test_takes_from_filenames_on_missing_dir_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "nao_existe"
    assert takes_from_filenames(missing, "sentando") == set()


def test_takes_from_filenames_parses_take_numbers(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "sentando"
    scenario_dir.mkdir()
    for name in ["sentando_take01_p0.npy", "sentando_take02_p0.npy",
                 "sentando_take02_p1.npy"]:
        np.save(scenario_dir / name, np.zeros((1, 17, 3)))

    assert takes_from_filenames(scenario_dir, "sentando") == {1, 2}


def test_takes_from_filenames_ignores_labels_files(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "sentando"
    scenario_dir.mkdir()
    np.save(scenario_dir / "sentando_take01_p0.npy", np.zeros((1, 17, 3)))
    np.save(scenario_dir / "sentando_take01_p0_labels.npy", np.zeros(1))

    # Sem isso, o take 01 seria contado a partir de 2 arquivos diferentes,
    # mas o número em si (1) é o mesmo — o teste real de "não conta em dobro"
    # está em test_takes_from_filenames_parses_take_numbers (p0 e p1 do take02
    # produzem o MESMO número 2, não dois números diferentes).
    assert takes_from_filenames(scenario_dir, "sentando") == {1}


def test_takes_from_filenames_ignores_other_scenarios(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "misto"
    scenario_dir.mkdir()
    np.save(scenario_dir / "sentando_take01_p0.npy", np.zeros((1, 17, 3)))
    np.save(scenario_dir / "deitando_take01_p0.npy", np.zeros((1, 17, 3)))

    assert takes_from_filenames(scenario_dir, "sentando") == {1}
