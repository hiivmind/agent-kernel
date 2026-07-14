import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).parents[2]


def test_wheel_includes_pep561_marker(tmp_path: Path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(tmp_path),
            str(ROOT),
        ],
        check=True,
    )
    wheel = next(tmp_path.glob("agent_kernel-*.whl"))

    with ZipFile(wheel) as archive:
        assert "agent_kernel/py.typed" in archive.namelist()
