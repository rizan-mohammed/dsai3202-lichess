# Databricks notebook source
# MAGIC %md
# MAGIC # Task 3 of 4 · Validate before publication
# MAGIC This notebook is the third task of your job. It runs only if **prepare** succeeded.
# MAGIC 
# MAGIC **Its job:** decide whether this run's data may be published. A profile *describes* data; a **quality gate** *decides*.
# MAGIC 
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Reads** | This run's `input.json` and input copy (saved by prepare). |
# MAGIC | **Writes** | Games that are ineligible for the 20-ply prediction to `processed/lichess_runs/<run_id>/quarantine/`, and the verdict to `curated/quality/lichess_runs/<run_id>/gate.json`. |
# MAGIC | **Stops when** | Any rule with policy **fail** is broken, or too many games are quarantined. The rules are listed in `lichess_00_common`, section 8. |
# MAGIC 
# MAGIC **Expected output in the job run:** `lichess_00_common loaded: ...`, the gate as JSON, then `QUALITY GATE PASSED`, or an error `QUALITY GATE FAILED: <rule names>`.

# COMMAND ----------

# MAGIC %run ./lichess_00_common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Load this run's saved input
# MAGIC Validate reads the copy that prepare saved, not the live silver table. So it checks exactly the data that publish will use. The manifest must belong to this run.

# COMMAND ----------

c = config()
manifest = read_json(f"{c['quality']}/input.json")
assert manifest["run_id"] == c["run_id"]
silver = spark.read.parquet(manifest["input_path"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Apply the gate and set ineligible games aside
# MAGIC `evaluate_gate` (from `lichess_00_common`) checks every rule and returns the evidence, the verdict and the games that cannot be used for the 20-ply prediction.
# MAGIC 
# MAGIC Those games are not deleted. They are saved to **quarantine** with their reason (`outside_month`, `untimed_game`, `too_short` or `incomplete_clocks`), so they can be inspected later. The saved quarantine is read back and counted.

# COMMAND ----------

gate, ineligible = evaluate_gate(silver, manifest["expected_rows"], c["month"])
gate["run_id"] = c["run_id"]
gate["quarantine_path"] = f"{c['work']}/quarantine"
if ineligible is not None and gate["quarantined_rows"]:
    ineligible.coalesce(8).write.mode("error").parquet(gate["quarantine_path"])
    assert spark.read.parquet(gate["quarantine_path"]).count() == gate["quarantined_rows"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Save the verdict, then pass or fail the task
# MAGIC The order matters:
# MAGIC 1. **Save `gate.json` first.** If the gate fails, the evidence of *why* still exists.
# MAGIC 2. **Then raise an error if the gate failed.** The job scheduler only sees whether a task succeeded or failed, not what it printed. A failed task is what stops publish, because publish uses `Run if = All succeeded`. Printing a warning alone would let publish run.

# COMMAND ----------

write_json(f"{c['quality']}/gate.json", gate)
print(json.dumps(gate, indent=2))
if not gate["passed"]:
    raise ValueError("QUALITY GATE FAILED: " + ", ".join(gate["failed_rules"]))
print("QUALITY GATE PASSED")
