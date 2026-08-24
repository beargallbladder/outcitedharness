LIVE function from lib/db/cr-artifacts.ts. House rule 6.

Rows: hero_rollup 2026-W24, 2026-W28, 2026-W31; recall_pack 2026-W30.
Return EXACT JSON: {"a": ..., "b": ..., "c": ..., "d": ...}
a = findArtifactWeekAsOf('hero_rollup', '2026-W30')
b = findArtifactWeekAsOf('hero_rollup', '2026-W31')
c = findArtifactWeekAsOf('hero_rollup', '2026-W23')
d = findArtifactWeekAsOf('recall_pack', '2026-W52')
Use week strings or null. JSON only.
