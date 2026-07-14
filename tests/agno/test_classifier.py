from types import SimpleNamespace

from agent_kernel.core.intents import Intent
from agent_kernel.integrations.agno import AgnoClassifier


class ClassifiedIntent(Intent):
    mode: str = "safe"


class FakeRegistry:
    intent_type = ClassifiedIntent


class FakeAgent:
    def __init__(self, response):
        self.response = response
        self.run_calls = []

    def run(self, message, **kwargs):
        self.run_calls.append((message, kwargs))
        return self.response


class FakeAgentFactory:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.agents = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        agent = FakeAgent(self.response)
        self.agents.append(agent)
        return agent

    @property
    def last_kwargs(self):
        return self.calls[-1]


def test_classifier_constructs_tool_free_structured_agent():
    model = object()
    instructions = ["Choose the best registered action."]
    factory = FakeAgentFactory(SimpleNamespace(content=None))

    AgnoClassifier(
        registry=FakeRegistry(),
        model=model,
        instructions=instructions,
        agent_factory=factory,
    )

    assert factory.last_kwargs["model"] is model
    assert factory.last_kwargs["instructions"] == instructions
    assert factory.last_kwargs["tools"] == []
    assert factory.last_kwargs["output_schema"] is ClassifiedIntent


def test_classifier_runs_message_and_normalizes_response_content():
    response = SimpleNamespace(
        content={
            "action": "spin",
            "confidence": 0.9,
            "brief": "spin the wheel",
            "mode": "fast",
        }
    )
    factory = FakeAgentFactory(response)
    classifier = AgnoClassifier(
        registry=FakeRegistry(),
        model=object(),
        instructions=[],
        agent_factory=factory,
    )

    intent = classifier.classify("please spin")

    assert intent == ClassifiedIntent(
        action="spin",
        confidence=0.9,
        brief="spin the wheel",
        mode="fast",
    )
    assert factory.agents[0].run_calls == [("please spin", {})]
