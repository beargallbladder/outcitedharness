Two LIVE defects, same house rule 10.

1) The ingest skips when kind+week is already in cr_artifacts.
2) catalogue-series-count.ts readFileSyncs the path verifyReleaseArtifact
   returned. records.jsonl is 109 MB; serverless bundles only the .gz twin.
   Local disk has both files.

Diagnose both. Name the correct checks (sha, not presence; prefer raw then
gz). State why prod 503s and local works.
