"""credit-scoring-ml-observability: reliability layer for a credit-scoring model."""

import os

# Privacy by default: no usage telemetry leaves the machine; silence MLflow agent hints.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "1.0.0"
