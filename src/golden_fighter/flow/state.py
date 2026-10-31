"""The Flow's state: everything a run decides and measures, in one pydantic model.

run_summary.json is this state written to disk, so the summary always matches what the Flow did.
"""
from pydantic import BaseModel, Field

# Run status -> CLI exit code. A validation failure (bad data) is told apart from a crew/API/setup error.
EXIT_CODES = {"succeeded": 0, "validation_failed": 1, "error": 2}


class StageRecord(BaseModel):
    name: str                       # preflight | analyst | validation_gate | scientist
    status: str                     # ok | skipped | validation_failed | error
    seconds: float = 0.0
    detail: str = ""


class FlowState(BaseModel):
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    mode: str = "llm"               # llm | no_llm (deterministic pipelines) ; plus from_artifacts / demo flags
    from_artifacts: bool = False
    demo_scenario: str | None = None
    model: str | None = None
    status: str = "running"         # running | succeeded | validation_failed | error
    exit_code: int | None = None
    failed_stage: str | None = None
    failure_reason: str | None = None
    validation_issues: list[str] = Field(default_factory=list)
    stages: list[StageRecord] = Field(default_factory=list)
    gate: dict | None = None
    crews: dict[str, dict] = Field(default_factory=dict)
    totals: dict = Field(default_factory=dict)
    artifacts: dict[str, list[dict]] = Field(default_factory=dict)
    metrics: dict | None = None
    inputs: dict = Field(default_factory=dict)
    paths: dict[str, str] = Field(default_factory=dict)
    glossary: dict[str, str] = Field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status in ("validation_failed", "error")
