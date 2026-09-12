from pathlib import Path

import numpy as np

from capture_app.recording import Detection, RecordingSession


def test_single_person_produces_one_clip(
    app_config, adl_scenario, make_keypoints, tmp_path: Path,
) -> None:
    session = RecordingSession(adl_scenario, take=1, config=app_config)
    kp = make_keypoints(1)
    for _ in range(5):
        session.add_frame([Detection(person_id=7, keypoints=kp)])

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    results = session.finalize(out_dir)

    assert len(results) == 1
    clip = results[0]
    assert clip.clip_id == "sentando_take01_p0"
    assert clip.n_frames == 5

    saved_kp = np.load(clip.keypoints_path)
    assert saved_kp.shape == (5, 17, 3)
    np.testing.assert_array_equal(saved_kp[0], kp)


def test_labels_are_zero_for_adl_and_two_for_fall(
    app_config, adl_scenario, fall_scenario, make_keypoints, tmp_path: Path,
) -> None:
    kp = make_keypoints()

    adl_session = RecordingSession(adl_scenario, take=1, config=app_config)
    for _ in range(3):
        adl_session.add_frame([Detection(person_id=1, keypoints=kp)])
    adl_dir = tmp_path / "adl"
    adl_dir.mkdir()
    adl_clip = adl_session.finalize(adl_dir)[0]
    assert set(np.load(adl_clip.labels_path).tolist()) == {0}

    fall_session = RecordingSession(fall_scenario, take=1, config=app_config)
    for _ in range(3):
        fall_session.add_frame([Detection(person_id=1, keypoints=kp)])
    fall_dir = tmp_path / "fall"
    fall_dir.mkdir()
    fall_clip = fall_session.finalize(fall_dir)[0]
    assert set(np.load(fall_clip.labels_path).tolist()) == {2}


def test_missing_frames_for_a_person_are_zero_filled(
    app_config, adl_scenario, make_keypoints, tmp_path: Path,
) -> None:
    session = RecordingSession(adl_scenario, take=1, config=app_config)
    kp = make_keypoints()
    session.add_frame([Detection(person_id=1, keypoints=kp)])
    session.add_frame([])                                   # pessoa some no frame 1
    session.add_frame([Detection(person_id=1, keypoints=kp)])

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    clip = session.finalize(out_dir)[0]

    saved = np.load(clip.keypoints_path)
    assert saved.shape == (3, 17, 3)
    np.testing.assert_array_equal(saved[0], kp)
    np.testing.assert_array_equal(saved[1], np.zeros((17, 3)))
    np.testing.assert_array_equal(saved[2], kp)


def test_two_people_in_frame_produce_two_clips(
    app_config, adl_scenario, make_keypoints, tmp_path: Path,
) -> None:
    session = RecordingSession(adl_scenario, take=1, config=app_config)
    kp_a, kp_b = make_keypoints(1), make_keypoints(2)
    for _ in range(4):
        session.add_frame([
            Detection(person_id=10, keypoints=kp_a),
            Detection(person_id=20, keypoints=kp_b),
        ])

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    results = session.finalize(out_dir)

    assert {r.clip_id for r in results} == {
        "sentando_take01_p0", "sentando_take01_p1",
    }


def test_person_below_min_frames_is_discarded(
    app_config, adl_scenario, make_keypoints, tmp_path: Path,
) -> None:
    # app_config fixture usa min_frames_per_person=2
    session = RecordingSession(adl_scenario, take=1, config=app_config)
    kp = make_keypoints()
    session.add_frame([Detection(person_id=1, keypoints=kp)])   # só 1 frame válido
    session.add_frame([])
    session.add_frame([])

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    results = session.finalize(out_dir)

    assert results == []


def test_no_detections_at_all_produces_no_clips(
    app_config, adl_scenario, tmp_path: Path,
) -> None:
    session = RecordingSession(adl_scenario, take=1, config=app_config)
    session.add_frame([])
    session.add_frame([])

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert session.finalize(out_dir) == []


def test_clip_id_encodes_scenario_take_and_person_idx(
    app_config, adl_scenario, make_keypoints, tmp_path: Path,
) -> None:
    session = RecordingSession(adl_scenario, take=7, config=app_config)
    kp = make_keypoints()
    for _ in range(3):
        session.add_frame([Detection(person_id=1, keypoints=kp)])

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    clip = session.finalize(out_dir)[0]

    assert clip.clip_id == "sentando_take07_p0"
