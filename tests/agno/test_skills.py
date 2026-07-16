import json
from pathlib import Path
from unittest.mock import patch

from agent_kernel.integrations.agno.skills import AgnoSkillSource, SafeSkills


def write_skill(tmp_path: Path, *, name: str, script: str) -> Path:
    source = tmp_path / name
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill\n---\nRead status safely.\n",
        encoding="utf-8",
    )
    (scripts / "probe.py").write_text(script, encoding="utf-8")
    return source


def test_safe_skills_exposes_read_only_access(tmp_path):
    source = write_skill(tmp_path, name="status", script="print('unsafe')")
    skills = SafeSkills.from_source(AgnoSkillSource("status", str(source)))
    tools = {tool.name: tool for tool in skills.get_tools()}
    assert set(tools) == {
        "get_skill_instructions",
        "get_skill_reference",
        "get_skill_script_source",
    }
    assert "execute" not in tools["get_skill_script_source"].parameters["properties"]


def test_safe_skills_reads_script_without_executing_it(tmp_path):
    source = write_skill(tmp_path, name="status", script="print('unsafe')")
    skills = SafeSkills.from_source(AgnoSkillSource("status", str(source)))
    script_reader = {
        tool.name: tool.entrypoint for tool in skills.get_tools()
    }["get_skill_script_source"]

    with patch(
        "agno.skills.agent_skills.run_script",
        side_effect=AssertionError("script execution crossed the safety boundary"),
    ) as run_script:
        result = script_reader(skill_name="status", script_path="probe.py")

    assert json.loads(result)["content"] == "print('unsafe')"
    run_script.assert_not_called()
