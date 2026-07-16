from dataclasses import dataclass
from typing import Protocol

from agno.skills import LocalSkills, Skills
from agno.tools.function import Function


@dataclass(frozen=True)
class AgnoSkillSource:
    id: str
    source_path: str


class AgnoSkillProvider(Protocol):
    def require(self, skill_id: str) -> AgnoSkillSource:
        raise NotImplementedError


class SafeSkills(Skills):
    @classmethod
    def from_source(cls, source: AgnoSkillSource) -> "SafeSkills":
        return cls([LocalSkills(source.source_path)])

    def _get_skill_script_source(self, skill_name: str, script_path: str) -> str:
        return super()._get_skill_script(skill_name, script_path, execute=False)

    def get_tools(self) -> list[Function]:
        return [
            Function(
                name="get_skill_instructions",
                description="Load the selected Skill instructions.",
                entrypoint=self._get_skill_instructions,
            ),
            Function(
                name="get_skill_reference",
                description="Read a reference bundled with the selected Skill.",
                entrypoint=self._get_skill_reference,
            ),
            Function(
                name="get_skill_script_source",
                description="Read, but never execute, a script bundled with the selected Skill.",
                entrypoint=self._get_skill_script_source,
            ),
        ]
