import pytest

from desktop_agent.protocol import ProtocolError, validate_action, validate_actions


def test_validate_move_ok():
    a = validate_action({"op": "move", "x": 1, "y": 2})
    assert a["op"] == "move"


def test_validate_move_missing_fields():
    with pytest.raises(ProtocolError):
        validate_action({"op": "move", "x": 1})


def test_validate_click_default_button_ok():
    a = validate_action({"op": "click"})
    assert a["op"] == "click"


def test_validate_click_invalid_button():
    with pytest.raises(ProtocolError):
        validate_action({"op": "click", "button": "banana"})


def test_validate_key_combo_requires_non_empty_list():
    with pytest.raises(ProtocolError):
        validate_action({"op": "key_combo", "keys": []})


def test_validate_actions_list_ok():
    acts = validate_actions([
        {"op": "move", "x": 1, "y": 2},
        {"op": "type", "text": "hi", "delay": 0},
    ])
    assert len(acts) == 2


def test_validate_unknown_op():
    with pytest.raises(ProtocolError):
        validate_action({"op": "nope"})


def test_validate_move_delta_ok():
    a = validate_action({"op": "move_delta", "dx": 5, "dy": -7})
    assert a["op"] == "move_delta"


def test_validate_move_delta_missing_fields():
    with pytest.raises(ProtocolError):
        validate_action({"op": "move_delta", "dx": 1})
