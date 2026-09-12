from pathlib import Path

import pytest

from capture_app.scenario_catalog import ScenarioCatalogError, by_id, load_catalog


def write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "scenarios.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_valid_catalog(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, """
scenarios:
  - id: sentando
    label: ADL
    description: "Sentar em uma cadeira"
  - id: queda_lado_esquerdo
    label: FALL
""")
    scenarios = load_catalog(path)

    assert [s.id for s in scenarios] == ["sentando", "queda_lado_esquerdo"]
    assert scenarios[0].label == "ADL"
    assert scenarios[0].is_fall is False
    assert scenarios[1].is_fall is True
    assert scenarios[0].description == "Sentar em uma cadeira"


def test_description_defaults_to_empty_string(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, """
scenarios:
  - id: andando
    label: ADL
""")
    scenarios = load_catalog(path)
    assert scenarios[0].description == ""


def test_empty_catalog_raises(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "scenarios: []\n")
    with pytest.raises(ScenarioCatalogError, match="Nenhum cenário"):
        load_catalog(path)


def test_missing_scenarios_key_raises(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "outra_coisa: 1\n")
    with pytest.raises(ScenarioCatalogError, match="Nenhum cenário"):
        load_catalog(path)


def test_duplicate_id_raises(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, """
scenarios:
  - id: sentando
    label: ADL
  - id: sentando
    label: FALL
""")
    with pytest.raises(ScenarioCatalogError, match="duplicado"):
        load_catalog(path)


@pytest.mark.parametrize("bad_id", ["Sentando", "sentado-1", "sentado 1", "sentado!", ""])
def test_invalid_slug_raises(tmp_path: Path, bad_id: str) -> None:
    path = write_yaml(tmp_path, f"""
scenarios:
  - id: "{bad_id}"
    label: ADL
""")
    with pytest.raises(ScenarioCatalogError):
        load_catalog(path)


def test_invalid_label_raises(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, """
scenarios:
  - id: sentando
    label: MAYBE
""")
    with pytest.raises(ScenarioCatalogError, match="label"):
        load_catalog(path)


def test_by_id_builds_lookup(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, """
scenarios:
  - id: sentando
    label: ADL
  - id: queda_lado_esquerdo
    label: FALL
""")
    scenarios = load_catalog(path)
    index = by_id(scenarios)

    assert set(index) == {"sentando", "queda_lado_esquerdo"}
    assert index["sentando"].label == "ADL"
