from datetime import datetime

from capture_app.session_ids import new_session_id


def test_new_session_id_format_is_deterministic_under_fixed_clock() -> None:
    fixed_now = datetime(2026, 9, 12, 14, 32, 5)
    assert new_session_id(fixed_now) == "20260912_143205"


def test_new_session_id_defaults_to_now_and_is_a_string() -> None:
    session_id = new_session_id()
    assert isinstance(session_id, str)
    assert len(session_id) == len("20260912_143205")
