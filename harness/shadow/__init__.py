"""Opt-in, local-first shadow coding capture and execution."""

from harness.shadow.hook import capture_hook_event
from harness.shadow.comparison import admit_comparison, compare_task
from harness.shadow.models import ModelRuntime, ShadowPolicy, ShadowTask
from harness.shadow.processor import process_task
from harness.shadow.runner import run_one
from harness.shadow.spool import ShadowSpool

__all__ = [
    "ModelRuntime",
    "ShadowPolicy",
    "ShadowSpool",
    "ShadowTask",
    "admit_comparison",
    "capture_hook_event",
    "compare_task",
    "process_task",
    "run_one",
]
