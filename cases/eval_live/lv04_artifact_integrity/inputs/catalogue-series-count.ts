// lib/designwins/facade/catalogue-series-count.ts
// Isolated so family/board routes do not pull the 109 MB corpus
// into their serverless function graph.
import { readFileSync } from 'fs'
const path = verifyReleaseArtifact(loaded, recordsPath(loaded, aisle), pin.records_sha256)
records = readFileSync(path, 'utf8')  // ENOENT in prod if only the .gz twin shipped
