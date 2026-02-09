from __future__ import annotations

from desktop_agent.tools import submodel_batch_tool_spec, submodel_close_tool_spec, submodel_list_tool_spec


def test_submodel_tool_specs_shape() -> None:
    for spec in (submodel_batch_tool_spec(), submodel_list_tool_spec(), submodel_close_tool_spec()):
        assert spec["type"] == "function"
        assert isinstance(spec["name"], str) and spec["name"]
        assert isinstance(spec["parameters"], dict)

