# Agno upstream issue reports

These reports describe behavior observed with Agno 2.7.3. Neither issue blocks
`agent-kernel`: the integration uses a source-only Skill script reader and
keeps authorization in application-owned capability bindings.

## Allow applications to control or disable Skill script execution

**Suggested title:** Allow an application-controlled or disableable Agent
Skill script executor

### Current behavior

`agno.skills.Skills.get_tools()` exposes `get_skill_script`, whose
model-callable `execute=True` path invokes Agno's imported `run_script` helper.
Applications can neither omit execution while retaining source access nor
inject an executor that applies their sandbox, approval, audit, and identity
policies.

### Minimal reproduction

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from agno.skills import LocalSkills, Skills


with TemporaryDirectory() as directory:
    skill = Path(directory) / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demonstrate execution\n---\nRun the demo.\n"
    )
    script = skill / "scripts" / "demo.py"
    script.write_text("#!/usr/bin/env python3\nprint('executed')\n")
    script.chmod(0o755)

    skills = Skills([LocalSkills(str(skill))])
    script_tool = next(
        tool for tool in skills.get_tools() if tool.name == "get_skill_script"
    )
    print(
        script_tool.entrypoint(
            skill_name="demo",
            script_path="demo.py",
            execute=True,
        )
    )
```

The tool starts a subprocess through Agno's built-in executor. Subclassing can
replace the entire tool surface, but there is no public constructor option to
disable execution or inject an application executor.

### Requested behavior

Please add either or both of these public options:

- a source-only mode that does not expose an `execute` argument; and
- an injected script-executor callback or protocol, with execution disabled by
  default when no executor is supplied.

This would let applications retain progressive Skill discovery while routing
execution through their own security boundary. This is not blocking for
`agent-kernel`; `SafeSkills` exposes a separate source-only reader and requires
applications to bind any executor as a governed tool.

## Accept the Agent Skills `allowed-tools` representation

**Suggested title:** Parse `allowed-tools` as specified by the Agent Skills
frontmatter format

### Current behavior

The Agent Skills specification defines `allowed-tools` as a space-delimited
string. Agno 2.7.3 validates it as a YAML list of strings. A portable Skill
using the specification representation therefore fails Agno validation.

### Minimal reproduction

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from agno.skills import LocalSkills


with TemporaryDirectory() as directory:
    skill = Path(directory) / "demo"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: demo\n"
        "description: Demonstrate allowed tools\n"
        "allowed-tools: Read Bash(git:*)\n"
        "---\n"
        "Inspect the repository.\n"
    )

    LocalSkills(str(skill)).load()
```

Agno raises `SkillValidationError` containing:
`Field 'allowed-tools' must be a list`.

### Requested behavior

Accept the specification's string representation and retain it losslessly (or
parse it into a documented structured form). For compatibility, Agno could
continue accepting its current list form with a deprecation path.

This is not blocking for `agent-kernel`: capability bindings remain the
enforced authority boundary, independent of Skill frontmatter. Applications
can omit `allowed-tools` until Agno accepts the portable representation.
