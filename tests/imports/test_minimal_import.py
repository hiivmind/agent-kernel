import subprocess
import sys


def test_base_import_does_not_load_optional_frameworks():
    script = """
import builtins
import sys

original_import = builtins.__import__


def block_optional_frameworks(name, *args, **kwargs):
    if name == "agno" or name.startswith("agno."):
        raise ImportError("Agno imports are blocked for this test")
    return original_import(name, *args, **kwargs)


builtins.__import__ = block_optional_frameworks

import agent_kernel
from agent_kernel import AuthorityScope, PrincipalResolver, ResourceAuthority

assert agent_kernel.__version__ == "0.1.0"
assert AuthorityScope
assert PrincipalResolver
assert ResourceAuthority
assert "agno" not in sys.modules
assert "dspy" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
    )
