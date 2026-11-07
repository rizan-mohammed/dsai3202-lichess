# Databricks notebook source
# MAGIC %md
# MAGIC # Task 4 of 4 · Publish accepted games
# MAGIC This notebook is the last task of your job. It runs only if **validate** succeeded.
# MAGIC 
# MAGIC **Its job:** keep the games that are eligible for the 20-ply prediction and add the label to build **gold**, build the **features**, and record the result.
# MAGIC 
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Reads** | This run's `input.json`, `gate.json` and input copy. |
# MAGIC | **Writes** | This run's gold (Delta) and features (Parquet) in `curated/lichess_runs/<run_id>/`; the current table `lichess_gold` and `curated/lichess_features_v1/`; and `coverage.json`, `comparison.json` and `success.json` in `curated/quality/lichess_runs/<run_id>/`. |
# MAGIC | **Stops when** | The saved gate did not pass, this is a test run, or the counts do not balance. |
# MAGIC 
# MAGIC **Two kinds of output.** Each run keeps its own snapshot that is never overwritten (for comparison and reproducibility). The *current* paths, `lichess_gold` and `lichess_features_v1`, always hold the latest published result.
# MAGIC 
# MAGIC **Expected output in the job run:** `lichess_00_common loaded: ...`, the success record as JSON, `BALANCED: ... silver = ... gold + ... quarantined`, and `PUBLISHED: <run_id>`.

# COMMAND ----------

# MAGIC %run ./lichess_00_common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Check the saved gate again
# MAGIC The job already runs publish only after validate succeeds. This notebook checks the saved verdict once more anyway: if someone changes the job settings by mistake, a failed or test run still cannot publish. Two independent safeguards are safer than one (**defence in depth**).

# COMMAND ----------

c = config()
manifest = read_json(f"{c['quality']}/input.json")
gate = read_json(f"{c['quality']}/gate.json")
assert gate["run_id"] == c["run_id"] and gate["passed"], "No passing gate for this run."
assert manifest["test_bad_data"] == "false", "Test runs never publish."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Build gold from the eligible games, and check the balance
# MAGIC Gold keeps only the silver games that are eligible for the 20-ply prediction (see `eligibility_reason` in `lichess_00_common`) and adds the target `label` (0 White win, 1 draw, 2 Black win).
# MAGIC 
# MAGIC Two checks prove that no game was lost or duplicated:
# MAGIC - gold rows = the rows that validate accepted;
# MAGIC - silver rows = gold rows + quarantined rows.

# COMMAND ----------

silver = spark.read.parquet(manifest["input_path"])
label = (F.when(F.col("result") == "1-0", 0).when(F.col("result") == "1/2-1/2", 1).otherwise(2).cast("int"))
gold = (silver.filter(eligibility_reason(c["month"]).isNull())
        .select(*SILVER_COLUMNS, label.alias("label")))
gold_rows = gold.count()
assert gold_rows > 0 and gold_rows == gate["accepted_rows"]
assert gate["rows_in"] == gold_rows + gate["quarantined_rows"], "Counts do not balance."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Save this run's gold and features
# MAGIC Both are written into this run's own folder with `mode("error")`, which fails if the folder already exists, so they are never overwritten. Gold is saved as Delta; the features are built by `make_features` (from `lichess_00_common`) and saved as Parquet.
# MAGIC 
# MAGIC Each output is read back after writing, and the features must have exactly the expected columns and one row per gold game.

# COMMAND ----------

gold_path, features_path = f"{c['out']}/gold", f"{c['out']}/features"
gold.coalesce(16).write.format("delta").mode("error").save(gold_path)
gold = spark.read.format("delta").load(gold_path)
features = make_features(gold)
assert features.columns == FEATURE_COLUMNS
features.coalesce(8).write.mode("error").parquet(features_path)
features = spark.read.parquet(features_path)
assert features.count() == gold_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Measure who was kept
# MAGIC `coverage` compares the number of games per result and per time-control class before the eligibility filter (silver) and after it (gold). A good overall total can hide one group that lost many games, so the result is saved per group in `coverage.json`, together with the label shares, how well the rating difference separates winners, and the funnel from bronze to gold.

# COMMAND ----------

white = (F.col("label") == 0).cast("double")
silver_x = silver.withColumn("speed_class", speed_class_expr())
gold_x = gold.withColumn("speed_class", speed_class_expr())
eda = {"population": "all silver games and eligible gold games; no sample",
    "results": [r.asDict() for r in coverage(silver_x, gold_x, "result").orderBy("result").collect()],
    "speed_classes": [r.asDict() for r in coverage(silver_x, gold_x, "speed_class").orderBy("speed_class").collect()],
    "label_counts": [r.asDict() for r in features.groupBy("label").count().orderBy("label").collect()],
    "white_win_share_by_speed_class": [r.asDict() for r in features.groupBy("speed_class").agg(
        F.count("*").alias("games"), F.avg(white).alias("white_win_share"),
        F.avg("elo_mean").alias("mean_elo")).orderBy("speed_class").collect()],
    "elo_diff_quantiles_1_25_50_75_99": features.approxQuantile("elo_diff", [.01, .25, .5, .75, .99], 0.01),
    "corr_elo_diff_white_win": features.agg(F.corr("elo_diff", white)).first()[0],
    "share_unequal_start_clock": features.agg(F.avg("unequal_start")).first()[0],
    "feature_null_counts": features.agg(*[F.sum(F.col(x).isNull().cast("long")).alias(x)
                                          for x in FEATURE_COLUMNS]).first().asDict(),
    "funnel": {"bronze_rows": manifest["silver_report"]["rows_in"], "silver_rows": gate["rows_in"],
               "quarantined_ineligible": gate["quarantined_rows"], "gold_rows": gold_rows,
               "silver_dropped_by_step": manifest["silver_report"]["dropped_by_step"]}}
write_json(f"{c['quality']}/coverage.json", eda)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 · Rerun comparison (only when `compare_to` is set)
# MAGIC A comparison run used the same saved input as the earlier run. Its gold and features must therefore contain exactly the same rows. `same_content` checks every value of every row; equal row counts alone would not prove it.
# MAGIC 
# MAGIC If anything differs, the task stops before it changes the current outputs. The result is saved in `comparison.json`.

# COMMAND ----------

# Compare the complete retained snapshots before changing current output paths.
comparison = {"baseline_run": c["compare_to"], "checked": bool(c["compare_to"])}
if c["compare_to"]:
    previous = read_json(f"{c['curated']}/quality/lichess_runs/{c['compare_to']}/success.json")
    comparison["same_gold"] = same_content(gold, spark.read.format("delta").load(previous["gold_path"]))
    comparison["same_features"] = same_content(features, spark.read.parquet(previous["features_path"]))
    assert comparison["same_gold"] and comparison["same_features"], "Rerun changed values."
    print("SAME CONTENT: gold and features match run", c["compare_to"])
write_json(f"{c['quality']}/comparison.json", comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 · Update the current outputs
# MAGIC `lichess_gold` (a Unity Catalog table) and `curated/lichess_features_v1/` always hold the latest published result.
# MAGIC 
# MAGIC The first successful run creates `lichess_gold`: it writes the files and registers the table in one step, which needs the **CREATE EXTERNAL TABLE** permission on your curated location (you already used it in Lab-6). Later runs replace its contents. Before replacing anything, the cell checks that the existing table is in the expected place and has the same schema. After writing, it checks that the current outputs match this run's snapshot exactly.

# COMMAND ----------

# Current aliases are full rebuilds. Never run two writers in this schema.
current_gold = f"{c['curated']}/lichess_gold"
current_features = f"{c['curated']}/lichess_features_v1"
if spark.catalog.tableExists(c["gold_table"]):
    detail = spark.sql(f"DESCRIBE DETAIL {c['gold_table']}").first()
    assert detail["format"] == "delta" and detail["location"].rstrip("/") == current_gold, \
        "lichess_gold points elsewhere. Ask the instructor before writing."
    assert detail["partitionColumns"] == [], "Unexpected gold partition layout."
    existing = spark.table(c["gold_table"])
    assert existing.schema.simpleString() == gold.schema.simpleString(), "Existing gold schema differs."
    gold.select(*existing.columns).coalesce(16).write.mode("overwrite").insertInto(c["gold_table"])
else:
    # One step writes the files and registers the table, so a missing permission stops before any file is written.
    gold.coalesce(16).write.format("delta").mode("error").option("path", current_gold).saveAsTable(c["gold_table"])
spark.sql(f"COMMENT ON TABLE {c['gold_table']} IS 'Lichess games eligible for the 20-ply prediction, with the target label. Model features are in curated/lichess_features_v1.'")
features.coalesce(8).write.mode("overwrite").parquet(current_features)
assert same_content(gold, spark.table(c["gold_table"]))
assert same_content(features, spark.read.parquet(current_features))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 · Write the success record last
# MAGIC `success.json` records the run: its gold and feature paths, row counts, versions, source hash and duration. It is written **last**, so its existence proves the whole task finished. If publish stops halfway, there is no `success.json`, and the partial output must not be used.
# MAGIC 
# MAGIC `lichess_latest.json` points to the latest published run.

# COMMAND ----------

gold_version = int(spark.sql(f"DESCRIBE HISTORY {c['gold_table']}").first()["version"])
finished = datetime.now(timezone.utc)
report = {"run_id": c["run_id"], "month": c["month"], "gold_table": c["gold_table"], "gold_version": gold_version,
    "gold_path": gold_path, "features_path": features_path, "gold_rows": gold_rows,
    "quarantined_rows": gate["quarantined_rows"], "silver_rows": gate["rows_in"],
    "features_rows": gold_rows, "features_columns": FEATURE_COLUMNS,
    "current_gold_path": current_gold, "current_features_path": current_features,
    "silver_version": manifest["silver_version"], "source_sha256": manifest["silver_report"].get("source_sha256"),
    "bronze_games_parsed": manifest["silver_report"]["bronze_manifest"]["games_parsed"],
    "comparison": comparison, "finished_at": finished.isoformat(),
    "pipeline_elapsed_seconds": (finished - datetime.fromisoformat(manifest["started_at"])).total_seconds()}
write_json(f"{c['quality']}/success.json", report)
write_json(f"{c['curated']}/quality/lichess_latest.json", report, overwrite=True)
print(json.dumps(report, indent=2))
print(f"BALANCED: {gate['rows_in']:,} silver = {gold_rows:,} gold + {gate['quarantined_rows']:,} quarantined")
print("PUBLISHED:", c["run_id"])
# Each write is separate. success.json and lichess_latest.json are written last.
# If publication fails midway, keep the old immutable run and start a fresh job run.
