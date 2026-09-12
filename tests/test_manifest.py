from pathlib import Path

from capture_app.manifest import (
    FIELDNAMES,
    ManifestEntry,
    append_entries,
    ensure_manifest,
    read_entries,
    takes_for_scenario,
)


def make_entry(**overrides) -> ManifestEntry:
    defaults = dict(
        clip_id="sentando_take01_p0",
        is_fall=False,
        src_fps=30.0,
        lbl_path=Path("/data/sentando_take01_p0_labels.npy"),
        group_id="sentando_take01",
        px_path=Path("/data/sentando_take01_p0.npy"),
        scenario_id="sentando",
        scenario_label="ADL",
        take_number=1,
        person_idx=0,
        n_frames=42,
        session_id="20260912_143205",
    )
    defaults.update(overrides)
    return ManifestEntry(**defaults)


def test_ensure_manifest_creates_file_with_header(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    ensure_manifest(path)

    assert path.exists()
    assert path.read_text(encoding="utf-8").strip() == ",".join(FIELDNAMES)


def test_ensure_manifest_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    ensure_manifest(path)
    append_entries(path, [make_entry()])

    ensure_manifest(path)  # não deve truncar o arquivo já existente

    assert len(read_entries(path)) == 1


def test_append_entries_writes_expected_row(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    append_entries(path, [make_entry()])

    rows = read_entries(path)
    assert len(rows) == 1
    row = rows[0]
    assert row["clip_id"] == "sentando_take01_p0"
    assert row["is_fall"] == "False"
    assert row["scenario_label"] == "ADL"
    assert row["take_number"] == "1"
    assert row["n_frames"] == "42"
    assert row["recorded_at"]  # preenchido automaticamente


def test_append_preserves_previous_rows(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    append_entries(path, [make_entry(clip_id="a_take01_p0")])
    append_entries(path, [make_entry(clip_id="b_take01_p0", scenario_id="b")])

    rows = read_entries(path)
    assert [r["clip_id"] for r in rows] == ["a_take01_p0", "b_take01_p0"]


def test_append_multiple_entries_in_one_call(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    append_entries(path, [
        make_entry(clip_id="sentando_take01_p0", person_idx=0),
        make_entry(clip_id="sentando_take01_p1", person_idx=1),
    ])

    rows = read_entries(path)
    assert [r["person_idx"] for r in rows] == ["0", "1"]


def test_read_entries_on_missing_file_returns_empty_list(tmp_path: Path) -> None:
    assert read_entries(tmp_path / "nao_existe.csv") == []


def test_takes_for_scenario_filters_by_scenario_id(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    append_entries(path, [
        make_entry(clip_id="sentando_take01_p0", scenario_id="sentando", take_number=1),
        make_entry(clip_id="sentando_take02_p0", scenario_id="sentando", take_number=2),
        make_entry(clip_id="deitando_take01_p0", scenario_id="deitando", take_number=1),
    ])

    assert takes_for_scenario(path, "sentando") == {1, 2}
    assert takes_for_scenario(path, "deitando") == {1}
    assert takes_for_scenario(path, "inexistente") == set()


def test_recorded_at_is_not_overwritten_when_explicitly_set() -> None:
    entry = make_entry(recorded_at="2026-01-01T00:00:00+00:00")
    assert entry.recorded_at == "2026-01-01T00:00:00+00:00"
