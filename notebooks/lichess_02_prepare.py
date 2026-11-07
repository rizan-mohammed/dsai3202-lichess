# Databricks notebook source
# MAGIC %md
# MAGIC # Task 2 of 4 · Prepare the input
# MAGIC This notebook is the second task of your job: **silver → prepare → validate → publish**. You do not run it by hand; the job runs it.
# MAGIC 
# MAGIC **Its job:** decide exactly which silver data this run will use, save a private copy of it, and describe it with a profile.
# MAGIC 
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Reads** | Your silver report (`curated/quality/lichess_drop_report.json`) and your silver table at the exact version in that report. |
# MAGIC | **Writes** | This run's input copy in `processed/lichess_runs/<run_id>/`, plus `input.json` and `profile.json` in `curated/quality/lichess_runs/<run_id>/`. |
# MAGIC | **Stops when** | The report does not match your table, the counts do not balance, or the columns have the wrong types. |
# MAGIC 
# MAGIC **Why save a private copy?** The silver table may change later. A run that keeps its own copy can always be checked and repeated with exactly the same input. Validate and publish read this copy, never the live table.
# MAGIC 
# MAGIC **Expected output in the job run:** `lichess_00_common loaded: ...` at the start, the profile as JSON, and finally `INPUT SAVED: <run_id> ...`.

# COMMAND ----------

# MAGIC %run ./lichess_00_common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Read the job parameters and check the silver report
# MAGIC `config()` (from `lichess_00_common`) reads the seven job parameters and builds this run's paths.
# MAGIC 
# MAGIC The checks make sure we start from a trustworthy handoff:
# MAGIC - The report must belong to **your** silver table and to this month.
# MAGIC - The silver task's own accounting must balance: rows in = rows kept + rows dropped by each cleaning step.
# MAGIC - A controlled-failure test needs a successful earlier run in `compare_to`, so it can later prove that the earlier output was not changed.

# COMMAND ----------

c = config()
started = datetime.now(timezone.utc).isoformat()
report = read_json(f"{c['curated']}/quality/lichess_drop_report.json")
assert report["table"] == c["table"], "The silver report belongs to another table."
assert report["month"] == c["month"], "The silver report belongs to another month."
assert report["rows_in"] == EXPECTED_INPUT_ROWS and report["rows_out"] > 0
assert report["rows_in"] == report["rows_out"] + sum(report["dropped_by_step"].values())
assert c["test_bad_data"] == "false" or c["compare_to"], "The test needs a successful baseline in compare_to."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Choose this run's input
# MAGIC There are two cases.
# MAGIC 
# MAGIC **Normal run (`compare_to` empty):** read silver with `VERSION AS OF` (Delta time travel) at the version recorded in your silver report. Even if silver changes tomorrow, this run uses the version that the silver task approved.
# MAGIC 
# MAGIC **Comparison run (`compare_to` set):** reuse the input that the earlier run saved. Same input + same code must give the same output. If the output differs, the code is not repeatable.
# MAGIC 
# MAGIC Finally, the column types and the row count are checked.

# COMMAND ----------

if c["compare_to"]:
    previous = read_json(f"{c['curated']}/quality/lichess_runs/{c['compare_to']}/success.json")
    base = read_json(f"{c['curated']}/quality/lichess_runs/{c['compare_to']}/input.json")
    assert previous["run_id"] == c["compare_to"]
    source = spark.read.parquet(base["input_path"])
    source_version, source_report = base["silver_version"], base["silver_report"]
    expected_rows = base["expected_rows"]
else:
    source_version, source_report = int(report["silver_version"]), report
    source = spark.sql(f"SELECT * FROM {c['table']} VERSION AS OF {source_version}")
    expected_rows = int(report["rows_out"])
assert not schema_errors(source), schema_errors(source)
assert source.count() == expected_rows, "Saved silver count differs from the silver report."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Controlled-failure test (only when `test_bad_data = true`)
# MAGIC For the test run only, one game's result is changed to `2-0`: a result that does not exist in chess.
# MAGIC 
# MAGIC The change is made in **this run's copy only**. Your silver table and earlier runs are never touched. The validate task should catch it and stop the job.
# MAGIC 
# MAGIC In a normal run this cell does nothing.

# COMMAND ----------

if c["test_bad_data"] == "true":
    chosen = source.orderBy(*KEY).first()
    source = source.withColumn("result", F.when(F.col("game_id") == F.lit(chosen["game_id"]),
                               F.lit("2-0")).otherwise(F.col("result")))
    print("TEST INPUT ONLY: one result changed to 2-0.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Save this run's input copy and its manifest
# MAGIC - The input is written with `mode("error")`: the write fails if the folder already exists, so a run can never overwrite another run's folder.
# MAGIC - The saved copy is read back and counted, to prove that the write is complete.
# MAGIC - `input.json` (the **manifest**) records where the input came from: table, version, paths, start time and the run's settings.
# MAGIC - For a test run, it also records what was published before the test, so `lichess_05_inspect` (6.1) can later prove that nothing changed.
# MAGIC 
# MAGIC If a run is interrupted, do not try to repair it: start a fresh **Run now**, which gets a new run ID and new folders.

# COMMAND ----------

input_path = f"{c['work']}/input"
source.select(*SILVER_COLUMNS).coalesce(16).write.mode("error").parquet(input_path)
saved = spark.read.parquet(input_path)
assert saved.count() == expected_rows
manifest = {"run_id": c["run_id"], "month": c["month"], "test_bad_data": c["test_bad_data"],
    "compare_to": c["compare_to"], "silver_table": c["table"], "silver_version": source_version,
    "silver_report": source_report, "expected_rows": expected_rows,
    "input_path": input_path, "started_at": started}
if c["test_bad_data"] == "true":
    manifest["publication_before_test"] = read_json(f"{c['curated']}/quality/lichess_latest.json")
    manifest["gold_version_before_test"] = int(spark.sql(f"DESCRIBE HISTORY {c['gold_table']}").first()["version"])
write_json(f"{c['quality']}/input.json", manifest)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 · Profile every input game
# MAGIC `profile` (from `lichess_00_common`) summarises all games, not a sample: missing values, game-time range, Elo ranges, plies percentiles, results, time controls and games per day. The summary is saved as `profile.json`.
# MAGIC 
# MAGIC A profile only **describes** the data. Deciding whether the data may be published is the job of the next task, validate.
# MAGIC 
# MAGIC **Expected output:** the profile as JSON, then `INPUT SAVED: <run_id> ...`.

# COMMAND ----------

summary = profile(saved)
write_json(f"{c['quality']}/profile.json", summary)
print(json.dumps(summary, indent=2, default=str))
print("INPUT SAVED:", c["run_id"], input_path)
