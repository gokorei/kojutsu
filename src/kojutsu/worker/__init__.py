"""The unattended capture loop.

A separate concern from capture itself, kept in its own package so the boundary is
visible in the tree rather than only in a docstring: the loop drives work and the
capture system records what happened, and neither reaches into the other's
internals.
"""

from .loop import (
    CAPTURE_ONLY_STEPS,
    DELEGATED_STEPS,
    CycleOutcome,
    CycleReport,
    CycleSteps,
    Step,
    Worker,
    WorkerConfig,
    WorkItem,
    WorkSource,
    backoff_for,
)
from .state import ClaimRecord, WorkerState
from .steps import CaptureCycleSteps, CaptureOnlyCycleSteps, ImplementerRequiredError

__all__ = [
    "CAPTURE_ONLY_STEPS",
    "DELEGATED_STEPS",
    "CaptureCycleSteps",
    "CaptureOnlyCycleSteps",
    "ClaimRecord",
    "CycleOutcome",
    "CycleReport",
    "CycleSteps",
    "ImplementerRequiredError",
    "Step",
    "WorkItem",
    "WorkSource",
    "Worker",
    "WorkerConfig",
    "WorkerState",
    "backoff_for",
]
