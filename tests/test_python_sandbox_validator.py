from __future__ import annotations


def test_python_sandbox_validator_blocks_obvious_danger() -> None:
    from desktop_agent.tools import _validate_python_sandbox_code  # noqa: WPS433

    errs = _validate_python_sandbox_code("import os\nprint(os.listdir('.'))\n")
    assert errs

    errs2 = _validate_python_sandbox_code("import subprocess\nsubprocess.run(['whoami'])\n")
    assert errs2


def test_python_sandbox_validator_allows_numpy() -> None:
    from desktop_agent.tools import _validate_python_sandbox_code  # noqa: WPS433

    errs = _validate_python_sandbox_code("import numpy as np\nprint(np.arange(3))\n")
    assert errs == []

