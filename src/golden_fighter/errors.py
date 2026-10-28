"""Project exceptions. Every validation failure carries a list of human-readable issues."""


class DataValidationError(Exception):
    """Raised when data or an artifact breaks an expected rule. The pipeline must stop."""

    def __init__(self, stage: str, issues: list[str]):
        self.stage = stage
        self.issues = list(issues)
        bullet_list = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(f"[{stage}] data validation failed:\n{bullet_list}")
