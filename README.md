# DSAI3202 Phase 1: Lichess game-outcome data pipeline on Azure

**Student:** Rizan Mohammed (60306369), individual project  
**Submitted release:** Git tag `phase1-submission` (the commit it points to is the submitted version)  
**Instructor access:** the instructor is added as a collaborator on this repository

## Project and hypothesis

**Problem.** Predict the result of a Lichess game (White win, draw, Black win) after the first **10 moves per player (20 plies)**, using only information available at that point.

**Target.** `label`: 0 = White wins (`1-0`), 1 = draw (`1/2-1/2`), 2 = Black wins (`0-1`).

**Hypothesis (Phase 0).** H1: a gradient-boosted classifier using ratings, time control, ECO/opening and early clock usage achieves lower multiclass log loss than a rating-only logistic regression on later, unseen games (improvement of at least 0.005 with a bootstrap 95% CI excluding 0). H0: these extra features give no measurable improvement.

**Baseline plan.** Majority class, multinomial logistic regression on `elo\_diff`, rating-only gradient boosting, then feature-group ablations. Chronological split (earlier games train, later games test) using `game\_ts`.

**Approved refinements.** None so far.

**\*\*Scope note.\*\* The Phase 0 secondary timing experiment (5, 10 and 15 moves per player) is not part of this project's deliverables. Bronze keeps the first 20 plies, so checkpoints beyond move 10 are not available. The main hypothesis at move 10 (20 plies) is unchanged.**

## Dataset and architecture

|Item|Value|
|-|-|
|Source|Lichess Open Database, standard rated games, **May 2017**: https://database.lichess.org/standard/lichess\_db\_standard\_rated\_2017-05.pgn.zst|
|Format|PGN compressed with Zstandard (`.pgn.zst`)|
|Compressed size|3,585,539,491 bytes (3.34 GiB)|
|Games|11,693,919 parsed (0 unparseable)|
|SHA-256|`c4b2e39a382fb8a9dc5e766bbae0201d123c5c0ccc27f6e852d9a69e6661ab2f`|
|Source ETag / Last-Modified|`"63657a0d-d5b6fda3"` / Fri, 04 Nov 2022 20:46:05 GMT|
|Ingestion mode|Batch, started manually; refresh strategy: one new monthly file per run (new versioned raw path and manifest)|
|Uncompressed size|Larger than the 3.34 GiB compressed file, so the roughly 1 GB requirement is met; all 11,693,919 games were ingested and profiled|
|Data access|Azure Storage account `amazonlake60306369` (resource group `rg-dsai3202-rizan-60306369`), containers `raw`, `processed`, `curated`; reading needs the Storage Blob Data Reader (or Contributor) role and `az login`. The data is not in Git; the source manifest with its hash is in `evidence/ingestion/source-manifest.json`|

### Architecture

```mermaid
flowchart LR
  L\[Lichess .pgn.zst] --> V\[Azure VM: Python streaming parser]
  V -->|AzCopy| RAW\[(ADLS Gen2 raw: source + bronze Parquet)]
  RAW -->|AzCopy, verified| PROC\[(processed: bronze\_staged, silver Delta, rejects, quarantine)]
  PROC --> DB\[Databricks Jobs + Spark: silver, prepare, validate, publish]
  DB --> CUR\[(curated: gold Delta, features, quality evidence)]
  DB --> UC\[Unity Catalog: lichess\_silver, lichess\_gold]
```

|Service|Role|
|-|-|
|Azure VM (Ubuntu)|Receives the 3.3 GiB file, hashes it, parses PGN in one streaming pass into Parquet|
|ADLS Gen2 (`amazonlake60306369`)|`raw`: source and bronze (as received/parsed). `processed`: silver and per-run work. `curated`: gold, features, evidence|
|AzCopy|Idempotent uploads (`--overwrite=false`)|
|Azure Databricks (Jobs, Spark)|The four-task ETL and quality gate|
|Unity Catalog|Registers `lichess\_silver` and `lichess\_gold`|

**Why bronze copy in processed?** Bronze is in `raw` as the lecture defines. The Databricks compute cannot read `raw` (`INSUFFICIENT\_PERMISSIONS`), so a verified copy sits in `processed/lichess/bronze\_staged/`, the same pattern as Lab-4 (raw to processed) then Lab-5.

### Storage layout

```
raw/lichess/standard\_rated/2017-05/   lichess\_db\_standard\_rated\_2017-05.pgn.zst, source-manifest.json
raw/lichess/bronze/month=2017-05/     part-00000..00058.parquet (59), \_bronze-manifest.json
processed/lichess/bronze\_staged/month=2017-05/   copy of bronze + \_source-manifest.json
processed/lichess\_silver/                         Delta (table lichess\_silver)
processed/lichess\_silver\_rejects/run\_id=<uuid>/   rejected games with reason and raw record
processed/lichess\_runs/<job run id>/              input copy, quarantine
curated/lichess\_gold/                             Delta (table lichess\_gold), current
curated/lichess\_features\_v1/                      Parquet, current
curated/lichess\_runs/<job run id>/                gold, features per run
curated/quality/lichess\_runs/<job run id>/        input.json profile.json gate.json coverage.json comparison.json success.json
curated/quality/lichess\_drop\_report.json          silver report
```

### Schema, keys and assumptions (silver / gold)

Key: `game\_id` (Lichess id from the `Site` header), unique per game (0 duplicates found).

|Column|Type|Meaning|
|-|-|-|
|game\_id|string|Lichess game id (key)|
|event|string|Event header (rated game type or tournament)|
|result|string|`1-0`, `0-1`, `1/2-1/2` (target source; never a feature)|
|utc\_date|string|Game date `YYYY.MM.DD`|
|game\_ts|timestamp|Start time (UTC) from UTCDate + UTCTime|
|white\_elo, black\_elo|int|Ratings before the game|
|white\_diff, black\_diff|int|Rating change after the game (leakage; never a feature)|
|eco, opening|string|Opening code and name|
|time\_control|string|`base+increment` seconds, or `-` for untimed|
|tc\_base, tc\_inc|int|Parsed time control|
|termination|string|How the game ended (leakage; never a feature)|
|n\_plies|int|Total plies in the game (leakage; never a feature)|
|moves20|array<string>|First 20 plies (SAN)|
|clocks20|array<int>|Seconds left after each of those plies|
|label (gold only)|int|0 White win, 1 draw, 2 Black win|

Assumptions: the monthly file also holds a few hours of the previous month (`outside\_month`, quarantined); a game's opening name is assigned by Lichess from the moves, so a few deep opening names may reflect moves beyond ply 20 (leakage risk, see below).

### Lineage

source `.pgn.zst` (sha256 in `source-manifest.json`) -> `parse\_lichess.py` -> bronze Parquet (`\_bronze-manifest.json`: 11,693,919 games) -> staged copy (byte sizes verified) -> `lichess\_silver` (drop report: 11,687,692 rows) -> run input copy (`input.json`) -> `lichess\_gold` + `features\_v1` (`success.json`). `success.json` carries `source\_sha256` and `bronze\_games\_parsed`.

## Run the pipeline

**Prerequisites.** Azure subscription with the lab resource group and lake; the Lab-2 VM; Databricks workspace with Unity Catalog (`dsai3202\_databricks.s\_60306369`); the instructor-granted data roles. `pip install -r requirements.txt` on the VM (`zstandard`, `pyarrow`).

**1. Ingest (VM).** Copy the `.zst` to the VM (`scp`), then:

```bash
mkdir -p \~/Lab-Phase1/{src,evidence,bronze}; cd \~/Lab-Phase1
sha256sum src/lichess\_db\_standard\_rated\_2017-05.pgn.zst | tee evidence/source.sha256
# write evidence/source-manifest.json (url, bytes, sha256, ETag, Last-Modified); then:
export AZCOPY\_AUTO\_LOGIN\_TYPE=AZCLI; source \~/.config/dsai3202/lab3.env
B="https://$SA.blob.core.windows.net/raw/lichess/standard\_rated/2017-05"
azcopy copy src/\*.zst "$B/" --overwrite=false ; azcopy copy evidence/source-manifest.json "$B/" --overwrite=false
python3 -m venv .venv \&\& .venv/bin/pip install -r requirements.txt
nohup .venv/bin/python src/ingest/parse\_lichess.py src/lichess\_db\_standard\_rated\_2017-05.pgn.zst bronze/month=2017-05 > evidence/parse.log 2>\&1 \&
# upload bronze to raw, then the staged copy to processed (same azcopy --overwrite=false), then verify sizes
```

Verification and retry evidence is in `evidence/ingestion/` (hash round-trip, receipts before/after, `SIZES MATCH`, `STAGED MATCHES RAW`).

**2. ETL (Databricks).** Import `notebooks/lichess\_\*.py` into one workspace folder. Run `lichess\_00\_common` once. Create a job with four notebook tasks on one compute, no retries, no schedule, concurrency 1:

|Task|Notebook|Depends on (Run if)|
|-|-|-|
|silver|`lichess\_01\_silver`|none|
|prepare|`lichess\_02\_prepare`|silver (all succeeded)|
|validate|`lichess\_03\_validate`|prepare (all succeeded)|
|publish|`lichess\_04\_publish`|validate (all succeeded)|

Job parameters: `lake=amazonlake60306369`, `catalog=dsai3202\_databricks`, `schema=s\_60306369`, `run\_id={{job.run\_id}}`, `month=2017-05`, `test\_bad\_data=false`, `compare\_to=` (empty). Exported configuration: `job/job-export.json`. Safe rerun: Run now with different parameters, `compare\_to=<successful run id>`. Controlled failure: `test\_bad\_data=true`.

**3. Inspect.** `lichess\_05\_inspect` reads the saved evidence (cells 3.1 to 6.1).

## Results and features

### Runs

|Purpose|Job run ID|Outcome|
|-|-|-|
|Success|`1022222200504518`|All four tasks passed; published|
|Safe rerun|`681959495558867`|`SAME CONTENT`: gold and features identical to the success run (11,129,109 rows)|
|Controlled failure|`1040472147064037`|`test\_bad\_data=true` builds a separate test input (a copy of the saved input with one result changed to `2-0`, inside this run only); gate failed on `result`; publish did not run; `BLOCKED SAFELY` (published gold version and contents unchanged)|
|Real gate failure (fixed)|`455376947516555`|Gate failed on `time\_control`: my rule rejected base 0 (`0+1`), which Lichess allows. Rule corrected; rerun passed|

Outputs of the success run: gold `curated/lichess\_runs/1022222200504518/gold` (Delta), features `.../features`, `lichess\_gold` version 1 after run 1 (each publish also writes a comment, so versions advance by 2 per run with identical content). `lichess\_silver` is at version 6: earlier versions are identical rewrites from reruns.

### Row accounting

|Stage|Rows|Share|
|-|-|-|
|Bronze (parsed games)|11,693,919|100%|
|Invalid: result not in {1-0, 0-1, 1/2-1/2}|6,227|0.053%|
|Invalid: game id, Elo, timestamp, time control; duplicates|0|0%|
|**Silver**|**11,687,692**|99.947%|
|Quarantined as ineligible for the 20-ply prediction|558,583|4.78% of silver|
|**Gold / features**|**11,129,109**|95.22% of silver|

Balance checks printed by the job: `11,693,919 = 11,687,692 + 6,227 + 0` (silver task) and `11,687,692 silver = 11,129,109 gold + 558,583 quarantined` (publish).

Quarantine reasons (rule `eligibility`, `gate.json` of run `1022222200504518`): `too\_short` 492,297 (4.21% of silver), `untimed\_game` 36,476 (0.31%), `outside\_month` 29,810 (0.26%), `incomplete\_clocks` 0. Total 558,583 (4.78%, tolerance 15%). Every game with at least 20 plies had a clock value for each of those plies, so clock coverage was complete. `coverage.json` counts 36,565 untimed games in silver; 89 of them are also outside the month and are counted once, under `outside\_month`, because each game gets the first reason that applies.

### Quality gate

Fail rules: schema, volume, keys, result, elo\_range, timestamp, time\_control, plies\_arrays, unique\_game. Quarantine rule: `eligibility` (at most 15% of silver). Warn rule: `clock\_gap` (at most 10%). A failed rule raises an error in `validate`, so `publish` (Run if = All succeeded) never runs. Silver keeps every valid game; ineligible games are never deleted, only set aside with a reason.

### Exploratory analysis (all games, no sample; `coverage.json`)

* **Classes:** White 5,502,211 (49.44%), Black 5,187,163 (46.61%), draw 439,735 (3.95%). The majority-class baseline is about 49.4% accuracy; the prior-only log loss is about 0.83 nats (computed from these shares).
* **Rating signal:** correlation between `elo\_diff` and a White win is 0.31. `elo\_diff` quantiles (1/25/50/75/99%): -1794, -75, 0, 73, 1787. The heavy tails are mismatched games; use clipping or robust handling.
* **Time controls:** bullet 3,212,374, blitz 5,294,001, rapid 2,437,078, classical 185,656 games in gold. Mean Elo falls from 1,719 (bullet) to 1,490 (classical). White win share is 49.2% to 49.7% in every class.
* **Who is lost in eligibility (retention):** draws 97.7% vs White wins 94.8% and Black wins 95.5%; bullet 96.7%, blitz 95.8%, rapid 93.7%, classical 91.0%. All 36,565 untimed games (`-`) are excluded. Slower games and decisive results lose more games, so the modelled population over-represents long, drawn-out games and fast time controls (selection bias, see risks).
* **Clock data:** 3.0% of games start with unequal clocks (berserk or odds); no feature has missing values.
* **Readiness:** no missing feature values; 11.1M rows is ample for gradient boosting; draws are rare (3.95%), so use log loss and class-aware evaluation, not accuracy. Expect the rating baseline to be strong, so the 0.005 log-loss threshold for H1 is demanding.

### Features (v1, saved to `curated/lichess\_features\_v1`)

19 columns, defined in `FEATURES.md` (definition, type, source, availability at prediction time): ratings and rating difference/mean, time control and speed class, ECO/opening family, first clocks, seconds used over the first 10 moves, clock-use difference, unequal-start flag, plus `game\_id`, `game\_ts` (for chronological splits) and the label.

**Leakage risks:** `white\_diff`, `black\_diff`, `termination`, `n\_plies` and `result` are never features (known only after the game). The opening name is assigned by Lichess from the moves and may include moves after ply 20 for deep lines; the Phase 2 ablation will test this. Eligibility requires 20 plies, which is known at move 10 in live use, but it removes early endings from training and evaluation. Only one month is used, so the chronological test set is the last days of May 2017.

## Resource use and contributions

**Cost/performance choice (measured).** Parse once on the VM with a streaming Zstandard reader instead of splitting games into small files or loading the file in Spark (`.zst` is not splittable). Measured: 6,100 games/s (200,000-game parts every \~32.5 s), about 32 minutes for 11.69M games, and the Parquet bronze is 736,859,702 bytes, 4.9 times smaller than the 3,585,539,491-byte source because only headers and the first 20 plies are kept. Publish took 198.7 s (`success.json`). A first silver run in the job was killed for memory (executor loss, code 52); I stopped caching in memory, built the raw-record JSON only for rejected rows, and raised shuffle partitions; the next run completed.

**Cost view (Azure Cost analysis, scope `rg-dsai3202-rizan-60306369`, 19 Oct to 6 Nov 2026, read on 6 Nov 2026):** actual cost USD 13.48 for the whole lab resource group since 19 Oct, covering all labs, not only Phase 1; forecast USD 14.21; no budget set. By service: Virtual Machines 8.52, Virtual Network (the VM's public IP) 2.22, Storage 1.51, Container Registry 1.18, Load Balancer 0.04, all others under 0.01. Everything is in Qatar Central. Databricks compute does not appear in this scope's view. The VM is the largest cost and runs only for ingestion, with daily auto-shutdown at 22:00 Qatar; the Standard public IP keeps billing while the VM is deallocated. The lake itself costs about USD 0.19.

**Cleanup.** Databricks compute terminated after the runs; VM auto-shutdown at 19:00 UTC (22:00 Qatar); lake and evidence kept for Phase 2; no resources deleted.

**Contributions.** Individual project: all work by Rizan Mohammed.

**Reused code and adaptations.** Lab-3 (AzCopy upload with `--overwrite=false`, manifests, receipts), Lab-5 `01\_silver` (contract, row accounting, Delta table, `SAME CONTENT`), Lab-6 (`00\_common`, `01\_prepare`, `02\_validate`, `03\_publish`, `04\_inspect`: job structure, saved input copy, quality gate with fail/quarantine/warn policies, `compare\_to`, controlled failure). Adaptations: PGN parser (own), Lichess contract and eligibility rules, 20-ply features, fingerprint comparison instead of `exceptAll` (11M rows with arrays), `result` corruption for the failure test, four-task job with the cleaner first.

## AI use

|Tool|What I asked|What it returned|What I changed|
|-|-|-|-|
|DeepSeek|I need you teach me how to re-work these notebooks to be adapted for my dataset (lichess) see attatched notebooks and sample dataset|Guide on Adapting Lab 6 Notebooks to Lichess Dataset - full response as pasted in AI_use .txt |Rewrote the notebooks for my dataset using this as a template or starting point |
