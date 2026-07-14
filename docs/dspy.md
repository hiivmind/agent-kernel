# DSPy classifiers and artifacts

Install on Python 3.12 or 3.13:

```bash
python -m pip install "agent-kernel[dspy]"
```

The initial DSPy matrix is Python 3.12 and 3.13. Core and Agno support Python
3.14, but DSPy is excluded there until the upstream DSPy/LiteLLM chain and its
PyO3-based dependencies support Python 3.14. Combined `agno,dspy` installs are
tested on 3.12 and 3.13 for the same reason.

```python
from pathlib import Path
import dspy

from agent_kernel.integrations.dspy import (
    DspyClassifier, build_signature, registry_fingerprint, save_artifact,
)
from spinner_domain import registry

module = dspy.Predict(build_signature(registry))
# Configure or optimize the module with DSPy before saving it.
path = Path("artifacts/spinner-classifier.json")
save_artifact(registry=registry, module=module, path=path)
print(registry_fingerprint(registry))

classifier = DspyClassifier.from_artifact(registry=registry, path=path)
```

The sibling metadata file contains the format version, exact program filename,
and SHA-256 fingerprint of the intent schema, actions, default/denied actions,
and bounded continuation authority vocabularies. Loading rejects mismatched
registries and altered paths before DSPy loads the program.

Fallback is explicit:

```python
from agent_kernel import Continuation


class ConservativeFallback:
    def classify(self, message: str, *, context: Continuation | None = None):
        del context
        return registry.intent_type(
            action=registry.default, confidence=1.0, brief=message,
        )


classifier = DspyClassifier.from_artifact(
    registry=registry,
    path=path,
    fallback=ConservativeFallback(),
)
```

With no fallback, module or normalization errors are raised.
