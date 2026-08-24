Production checkout (isolated). `scripts/ops/week-status.ts` is the weekly
truth-teller. A regression inverted MISSING VS reference: it now lists lanes
that are NEW in the current week (gemini-flash trap) instead of lanes that
were in the reference week and vanished.

Restore the correct set difference: missing = in reference, absent in current.
Do not invert it the other way. `run` until PASS, then finish.
