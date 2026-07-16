import subprocess
import sys


def test_base_import_does_not_load_optional_frameworks():
    script = """
import sys
import agent_kernel

assert agent_kernel.__version__ == "0.1.0"
assert "agno" not in sys.modules
assert not any(name.startswith("agent_kernel.integrations.agno") for name in sys.modules)
assert "dspy" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
    )
