"""Project-wide pytest configuration.

`freeze_time` vs. `langfuse` gotcha (found during Story 1.3, confirmed by
tech-lead 2026-07-22, hit again independently in Story 1.7/Task 1.7.2):
freezegun's module-attribute scan trips over a lazy `langfuse.api` Pydantic
import while time is frozen, and pydantic-core's schema generation chokes on
it. This only reproduces when the full suite runs (langfuse must already be
loaded by an earlier test module, e.g. `test_config.py`/
`test_langfuse_setup.py`, before `freeze_time` does its scan) -- running a
single `freeze_time`-using test file in isolation does not trigger it, which
is why it's easy to miss.

Story 1.3 worked around this per-test-file with `freeze_time(...,
ignore=["langfuse"])`. Configuring the ignore list once here, at collection
time, means every `freeze_time` call in the suite (present and future) is
covered without repeating the `ignore=` kwarg at each call site.
"""

import freezegun.config

freezegun.config.configure(default_ignore_list=["langfuse"])
