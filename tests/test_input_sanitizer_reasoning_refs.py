from desktop_agent.tools import _strip_status_field_from_input_items


def test_strip_reasoning_refs_from_message_items():
    items = [
        {"type": "reasoning", "id": "rs_1"},
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "reasoning": "rs_1",
            "content": [
                {"type": "output_text", "text": "hello", "reasoning_id": "rs_1"},
            ],
        },
    ]

    out = _strip_status_field_from_input_items(items)

    # Reasoning item dropped.
    assert all(not (isinstance(x, dict) and x.get("type") == "reasoning") for x in out)

    msg = next(x for x in out if isinstance(x, dict) and x.get("type") == "message")
    assert "status" not in msg
    assert "reasoning" not in msg
    assert "reasoning_id" not in str(msg)

