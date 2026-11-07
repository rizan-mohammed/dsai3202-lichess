# Databricks notebook source
# MAGIC %md
# MAGIC # Inspect your Lichess runs
# MAGIC This notebook reads the evidence that your job saved. It does not change any data. Do not add it as a job task.
# MAGIC 
# MAGIC **How to use it**
# MAGIC - Attach your running compute.
# MAGIC - Run the `%run ./lichess_00_common` cell first. It loads the shared rules, so this notebook can use them.
# MAGIC - Run each numbered cell when the guide asks for it. Type values only on the lines marked **YOUR VALUE**.

# COMMAND ----------

# MAGIC %run ./lichess_00_common

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.1 Use the shared rules from lichess_00_common
# MAGIC The cell above ran `lichess_00_common` inside this notebook. This cell proves it by calling `evaluate_gate`, the same function that the **validate** task runs on your real data. It checks four hand-made games (no lake data is used):
# MAGIC 
# MAGIC | Game | What is special |
# MAGIC |---|---|
# MAGIC | 1 | A correct game. |
# MAGIC | 2 | `result = "2-0"`: a result that does not exist. |
# MAGIC | 3 | `white_elo = 9999`: a rating above 4000. |
# MAGIC | 4 | An exact copy of game 1: the same `game_id`. |
# MAGIC 
# MAGIC **Before you run it, predict:** which rules will fail? **Expected result:** `result`, `elo_range` and `unique_game` fail, and the verdict is `QUALITY GATE FAILED`.

# COMMAND ----------

# 0.1 Try the shared quality gate on four hand-made games.
moves = ["e4", "e5"] * 10
clocks = list(range(60, 20, -2))
t = datetime(2017, 5, 10, 12, 0, 0)
def game(gid, result="1-0", welo=1500):
    return (gid, "Rated Blitz game", result, "2017.05.10", t, welo, 1500, 5, -5, "C20", "King's Pawn Game",
            "300+0", 300, 0, "Normal", 40, moves, clocks)
demo = spark.createDataFrame([game("g1"), game("g2", result="2-0"), game("g3", welo=9999), game("g1")],
    "game_id string, event string, result string, utc_date string, game_ts timestamp, white_elo int, "
    "black_elo int, white_diff int, black_diff int, eco string, opening string, time_control string, "
    "tc_base int, tc_inc int, termination string, n_plies int, moves20 array<string>, clocks20 array<int>")
demo_gate, _ = evaluate_gate(demo, expected_rows=4, month="2017-05")
display(spark.createDataFrame([(r["rule"], r["policy"], r["passed"], json.dumps(r["detail"]))
                               for r in demo_gate["rules"]], "rule string, policy string, passed boolean, detail string"))
print("QUALITY GATE PASSED" if demo_gate["passed"] else "QUALITY GATE FAILED: " + ", ".join(demo_gate["failed_rules"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Connect this notebook to your silver output
# MAGIC This notebook needs three names (`lake`, `catalog`, `schema`). They are prefilled with yours. The cell reads your silver report from your lake (`curated/quality/lichess_drop_report.json`), which must name `catalog.schema.lichess_silver`, so a wrong lake, catalog or schema is caught here.
# MAGIC 
# MAGIC **Expected result:** `READY:` with your silver row count and Delta version.

# COMMAND ----------

# 1.1 Connect to your silver output. Type the three names first (see the text above).
lake = "amazonlake60306369"      # YOUR VALUE
catalog = "dsai3202_databricks"  # YOUR VALUE
schema = "s_60306369"            # YOUR VALUE
lake, catalog, schema = lake.strip(), catalog.strip(), schema.strip()
assert re.fullmatch(r"[a-z0-9]{3,24}", lake), "Set lake: lowercase letters and digits only."
assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", catalog), "Set catalog."
assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema), "Set schema."
curated = f"abfss://curated@{lake}.dfs.core.windows.net"
try:
    report = read_json(f"{curated}/quality/lichess_drop_report.json")
except Exception as error:
    raise AssertionError("Cannot read curated/quality/lichess_drop_report.json. Run the silver task first, "
                         "and check the lake name.") from error
assert report["table"] == f"{catalog}.{schema}.lichess_silver", f"Your report names {report['table']}."
silver = spark.sql(f"SELECT * FROM {report['table']} VERSION AS OF {int(report['silver_version'])}")
assert not schema_errors(silver), schema_errors(silver)
assert silver.count() == report["rows_out"]
display(spark.sql(f"DESCRIBE DETAIL {report['table']}").select("format", "location"))
print("READY:", report["rows_out"], "silver rows; version", report["silver_version"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Read one job run's input and profile
# MAGIC Every click on **Run now** creates a job run with its own **job run ID**. The job saves that run's evidence under `curated/quality/lichess_runs/<job run ID>/`.
# MAGIC 
# MAGIC **Where to find the job run ID:** Jobs & Pipelines → your job → **Runs** tab → click the run → copy the **Job run ID** (not the Job ID, not a task run ID). Paste it on the line marked **YOUR VALUE**, then run the cell.
# MAGIC 
# MAGIC **Read it:** which result is the most common and what share is it? How many plies does the typical game have (`q50`)? Which days hold the games?

# COMMAND ----------

# 3.1 Read one job run's saved input and profile. Paste the job run ID first.
RUN_ID = "PASTE_JOB_RUN_ID"  # YOUR VALUE: the job run ID, for example "123456789012345"
runs_dir = f"{curated}/quality/lichess_runs"
runs = []
for f in dbutils.fs.ls(runs_dir):
    folder = f"{runs_dir}/{f.name.rstrip('/')}"
    names = [x.name for x in dbutils.fs.ls(folder)]
    if "input.json" not in names: continue
    m = read_json(f"{folder}/input.json")
    gate_state = (("passed" if read_json(f"{folder}/gate.json")["passed"] else "failed") if "gate.json" in names else "not checked")
    runs.append({"run_id": f.name.rstrip("/"), "started_at": m["started_at"], "test_bad_data": m["test_bad_data"],
                 "compare_to": m["compare_to"] or "-", "gate": gate_state, "published": str("success.json" in names)})
assert runs, "No job runs found yet. Run the job first, then run this cell again."
runs.sort(key=lambda r: r["started_at"])
display(spark.createDataFrame(runs))
run_id = RUN_ID.strip()
assert run_id in [r["run_id"] for r in runs], "That run is not in your lake. Did you copy the Job ID instead of the Job run ID?"
print("Inspecting run:", run_id)
quality = f"{runs_dir}/{run_id}"
manifest = read_json(f"{quality}/input.json")
summary = read_json(f"{quality}/profile.json")
print(json.dumps(summary, indent=2))
display(spark.createDataFrame(summary["results"]).orderBy("result"))
display(spark.createDataFrame(summary["days"]).orderBy("utc_date"))
print("All input games were profiled. Quantiles have at most 1% rank error.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Check the first successful run
# MAGIC Set `RUN_ID` in 3.1 to your **first successful** job run and run 3.1. Then run this cell.
# MAGIC 
# MAGIC **Expected result:** a passed gate, the success record, the coverage tables, the funnel and the first 10 gold games. `retained_pct` is the share of that group that survived the eligibility filter: a lower value in one group means the filter removed that group more than others, which can shift what a model learns (for example short bullet games).

# COMMAND ----------

# 4.1 Set RUN_ID in 3.1 to the first successful run and run 3.1 first.
gate = read_json(f"{quality}/gate.json")
success = read_json(f"{quality}/success.json")
eda = read_json(f"{quality}/coverage.json")
print(json.dumps(gate, indent=2))
print(json.dumps(success, indent=2))
display(spark.createDataFrame(eda["results"]).orderBy("result"))
display(spark.createDataFrame([{k: (-1 if v is None else v) for k, v in r.items()} for r in eda["speed_classes"]]).orderBy("speed_class"))
print("Label counts:", eda["label_counts"])
print("corr(elo_diff, white win):", eda["corr_elo_diff_white_win"])
print("Share of games with unequal start clocks:", eda["share_unequal_start_clock"])
for label, rows, column in [("result", eda["results"], "result"), ("time-control class", eda["speed_classes"], "speed_class")]:
    low = min(rows, key=lambda r: r["retained_pct"])
    print(f"Lowest retention by {label}: {low[column]} kept {low['retained_pct']}% ({low['after']:,} of {low['before']:,}).")
print(f"Balance: {gate['rows_in']:,} silver = {gate['accepted_rows']:,} gold + {gate['quarantined_rows']:,} quarantined.")
print("Funnel:", json.dumps(eda["funnel"]))
display(spark.read.format("delta").load(success["gold_path"]).orderBy(*KEY).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1 Prove that the rerun gave the same content
# MAGIC In 3.1, set `RUN_ID` to the **rerun** (the run you started with `compare_to`) and run 3.1. Then run this cell.
# MAGIC 
# MAGIC **Expected result:** `SAME CONTENT` for gold and features. The run IDs and the Delta version numbers differ (each run is a new event), but every value of every row is identical, because the same input and the same code gave the same result.

# COMMAND ----------

# 5.1 Set RUN_ID in 3.1 to the rerun and run 3.1 first.
comparison = read_json(f"{quality}/comparison.json")
assert comparison["checked"] and comparison["same_gold"] and comparison["same_features"]
print("SAME CONTENT:", json.dumps(comparison, indent=2))
baseline = read_json(f"{runs_dir}/{comparison['baseline_run']}/success.json")
rerun = read_json(f"{quality}/success.json")
print(f"\nRerun {run_id} reused the saved input of run {comparison['baseline_run']}.")
print(f"Gold rows: {baseline['gold_rows']:,} in the first run, {rerun['gold_rows']:,} in the rerun; every value matched.")
print(f"lichess_gold Delta version: {baseline['gold_version']} after the first run, {rerun['gold_version']} after the rerun: a new version with the same content.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 Prove that the bad-data test was blocked
# MAGIC In 3.1, set `RUN_ID` to the **failed test run** (started with `test_bad_data = true`) and run 3.1. Then run this cell. Skip 4.1 and 5.1.
# MAGIC 
# MAGIC **Expected result:** `BLOCKED SAFELY`: the gate failed on `result`, nothing was published, and the earlier gold and features are unchanged.

# COMMAND ----------

# 6.1 Set RUN_ID in 3.1 to the failed test run and run 3.1 first. Skip 4.1 and 5.1.
gate = read_json(f"{quality}/gate.json")
assert manifest["test_bad_data"] == "true"
assert not gate["passed"] and "result" in gate["failed_rules"]
before = manifest["publication_before_test"]
after = read_json(f"{curated}/quality/lichess_latest.json")
assert before == after, "The published run changed during the test."
current_version = int(spark.sql(f"DESCRIBE HISTORY {before['gold_table']}").first()["version"])
assert current_version == manifest["gold_version_before_test"]
assert same_content(spark.read.format("delta").load(before["gold_path"]), spark.table(before["gold_table"]))
assert same_content(spark.read.parquet(before["features_path"]), spark.read.parquet(before["current_features_path"]))
assert not any(f.name.rstrip("/") == run_id for f in dbutils.fs.ls(f"{curated}/lichess_runs"))
assert not any(f.name == "success.json" for f in dbutils.fs.ls(quality))
print(json.dumps(gate, indent=2))
print("BLOCKED SAFELY: no test publication; current gold version and feature contents unchanged.")
