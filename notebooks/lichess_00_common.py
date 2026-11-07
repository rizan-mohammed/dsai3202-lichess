# Databricks notebook source
# MAGIC %md
# MAGIC # Shared rules for the Lichess pipeline
# MAGIC This notebook holds everything that the Lichess job tasks must agree on: the **data contract**, the **quality-gate rules**, the **file locations** and the **feature definitions**. It is the Lichess version of the Lab-6 `00_common`.
# MAGIC 
# MAGIC **How it is used**
# MAGIC - You never run it as a job task. The task notebooks (`lichess_01_silver`, `lichess_02_prepare`, `lichess_03_validate`, `lichess_04_publish`) and the inspector (`lichess_05_inspect`) start with `%run ./lichess_00_common`.
# MAGIC - `lichess_01_silver` is self-contained, like Lab-5: it declares its own cleaning contract.
# MAGIC 
# MAGIC **Why one shared notebook?** One definition of every rule means prepare, validate and publish can never drift apart (*single source of truth*).
# MAGIC 
# MAGIC **Run this notebook once yourself** (Run all). It processes no data, so the only output is the last line, `lichess_00_common loaded: ...`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports

# COMMAND ----------

import json
import math
import re
from datetime import datetime, timezone
from pyspark.sql import functions as F, Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The data contract
# MAGIC These constants describe what a correct silver game looks like. Every check below is built from them.
# MAGIC 
# MAGIC | Constant | Meaning |
# MAGIC |---|---|
# MAGIC | `KEY` | One game is identified by its Lichess `game_id`. |
# MAGIC | `SILVER_COLUMNS` / `EXPECTED_TYPES` | The columns and exact types that `lichess_01_silver` must provide. `moves20` holds the first 20 plies, `clocks20` the clock (seconds left) after each of them. |
# MAGIC | `PLY_POINT` | The prediction point: 20 plies = 10 moves by each player. |
# MAGIC | `FEATURE_COLUMNS` | The columns saved for modelling (see `FEATURES.md`). |
# MAGIC | `ELIGIBILITY_TOLERANCE` | Silver keeps every valid game. Gold keeps only games that can be used for the prediction (in the month, timed, at least 20 plies, complete clocks). At most 15% of silver may be ineligible. |

# COMMAND ----------

KEY = ["game_id"]
PLY_POINT = 20
ELO_MIN, ELO_MAX = 100, 4000
RESULTS = ["1-0", "0-1", "1/2-1/2"]
EXPECTED_INPUT_ROWS = 11_693_919   # games parsed from lichess_db_standard_rated_2017-05.pgn.zst
ELIGIBILITY_TOLERANCE = 0.15       # Project policy: at most 15% of silver games may be ineligible.
CLOCK_GAP_WARN = 0.10
EXPECTED_TYPES = {"game_id": "string", "event": "string", "result": "string", "utc_date": "string",
    "game_ts": "timestamp", "white_elo": "int", "black_elo": "int", "white_diff": "int",
    "black_diff": "int", "eco": "string", "opening": "string", "time_control": "string",
    "tc_base": "int", "tc_inc": "int", "termination": "string", "n_plies": "int",
    "moves20": "array<string>", "clocks20": "array<int>"}
SILVER_COLUMNS = list(EXPECTED_TYPES)
GOLD_COLUMNS = SILVER_COLUMNS + ["label"]
FEATURE_COLUMNS = ["game_id", "game_ts", "label", "white_elo", "black_elo", "elo_diff", "elo_mean",
    "tc_base", "tc_inc", "speed_class", "eco", "eco_family", "opening_family",
    "w_clk_1", "b_clk_1", "w_used_10", "b_used_10", "used_diff_10", "unequal_start"]
# Known only after the game ends (or are the target): never model inputs.
LEAKAGE_COLUMNS = ["white_diff", "black_diff", "termination", "n_plies", "result", "moves20", "clocks20"]
assert not set(FEATURE_COLUMNS) & set(LEAKAGE_COLUMNS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Job parameters and file locations
# MAGIC `config()` reads the seven **job parameters** and builds every path from them.
# MAGIC 
# MAGIC | Parameter | Value you set on the job | Meaning |
# MAGIC |---|---|---|
# MAGIC | `lake` | `amazonlake<ID>` | Your ADLS Gen2 storage account. |
# MAGIC | `catalog` | `dsai3202_databricks` | The Unity Catalog catalog. |
# MAGIC | `schema` | `s_<ID>` | Your schema, which holds `lichess_silver`. |
# MAGIC | `run_id` | `{{job.run_id}}` | Databricks replaces this with the job run ID, so each run gets its own folders. |
# MAGIC | `month` | `2017-05` | The Lichess month being processed. |
# MAGIC | `test_bad_data` | `false` | `true` only for the controlled-failure test. |
# MAGIC | `compare_to` | empty | A successful earlier run ID, used for the rerun comparison. |
# MAGIC 
# MAGIC | Key | Location | Holds |
# MAGIC |---|---|---|
# MAGIC | `work` | `processed/lichess_runs/<run_id>/` | The run's saved copy of silver and the quarantined games. |
# MAGIC | `out` | `curated/lichess_runs/<run_id>/` | The run's gold data and features. |
# MAGIC | `quality` | `curated/quality/lichess_runs/<run_id>/` | The run's evidence: `input.json`, `profile.json`, `gate.json`, `coverage.json`, `comparison.json` (reruns only), `success.json`. |

# COMMAND ----------

def config():
    names = [("lake", ""), ("catalog", ""), ("schema", ""), ("run_id", ""), ("month", "2017-05"),
             ("test_bad_data", "false"), ("compare_to", "")]
    for name, default in names: dbutils.widgets.text(name, default)
    c = {name: dbutils.widgets.get(name).strip() for name, _ in names}
    assert re.fullmatch(r"[a-z0-9]{3,24}", c["lake"]), "Enter your lake name."
    for name in ["catalog", "schema"]:
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c[name]), "Enter your assigned catalog and schema."
    assert c["catalog"] != "hive_metastore", "Use the Unity Catalog."
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,80}", c["run_id"]), "Set run_id to the job run ID."
    assert re.fullmatch(r"\d{4}-\d{2}", c["month"]), "month looks like 2017-05"
    assert c["test_bad_data"] in ["true", "false"]
    assert not c["compare_to"] or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", c["compare_to"])
    assert c["compare_to"] != c["run_id"], "Compare with an earlier run."
    c["processed"] = f"abfss://processed@{c['lake']}.dfs.core.windows.net"
    c["curated"] = f"abfss://curated@{c['lake']}.dfs.core.windows.net"
    c["table"] = f"{c['catalog']}.{c['schema']}.lichess_silver"
    c["gold_table"] = f"{c['catalog']}.{c['schema']}.lichess_gold"
    c["work"] = f"{c['processed']}/lichess_runs/{c['run_id']}"
    c["out"] = f"{c['curated']}/lichess_runs/{c['run_id']}"
    c["quality"] = f"{c['curated']}/quality/lichess_runs/{c['run_id']}"
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    return c

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Saving and reading evidence
# MAGIC Each task saves its evidence as small JSON files in the lake. `write_json` refuses to overwrite an existing file unless asked, so a rerun cannot silently replace earlier evidence.

# COMMAND ----------

def read_json(path):
    return json.loads(dbutils.fs.head(path, 1024 * 1024))

def write_json(path, value, overwrite=False):
    dbutils.fs.mkdirs(path.rsplit("/", 1)[0])
    dbutils.fs.put(path, json.dumps(value, indent=2, default=str, allow_nan=False), overwrite)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Comparing data
# MAGIC - `schema_errors` lists every column that is missing or has the wrong type, compared with `EXPECTED_TYPES`.
# MAGIC - `fingerprint` and `same_content` decide whether two tables hold exactly the same rows. Each row of **every column** is hashed in two different ways and the hashes are summed, together with the row count. Order does not matter. Unlike the Lab-6 `exceptAll`, this does not need a full-row shuffle of 11 million games with arrays, which would take far too long. A changed value, a missing row or an extra duplicate changes the fingerprint.

# COMMAND ----------

def schema_errors(frame):
    actual = dict(frame.dtypes)
    return [f"{name}: expected {kind}, got {actual.get(name, 'MISSING')}"
            for name, kind in EXPECTED_TYPES.items() if actual.get(name) != kind]

def fingerprint(frame):
    cols = sorted(frame.columns)
    h1 = F.pmod(F.xxhash64(*cols), F.lit(2147483647)).cast("long")
    h2 = F.pmod(F.hash(*cols[::-1]), F.lit(2147483647)).cast("long")
    return tuple(frame.select(*cols).agg(F.count(F.lit(1)), F.sum(h1), F.sum(h2)).first())

def same_content(left, right):
    """Compare all rows of all columns. Order is irrelevant; duplicates count."""
    assert set(left.columns) == set(right.columns), "Column names differ."
    cols = sorted(left.columns)
    a, b = left.select(*cols), right.select(*cols)
    assert a.schema.simpleString() == b.schema.simpleString(), "Column types differ."
    return fingerprint(a) == fingerprint(b)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Profiling the input
# MAGIC `profile` summarises **all** input games (no sample): row count, missing values per column, first and last game time, Elo ranges, plies percentiles (q25, q50, q75, q95), results, the most common time controls and games per day. The percentiles use `approxQuantile` with a rank error of 0.01.

# COMMAND ----------

def profile(frame):
    n = frame.count()
    assert n > 0, "Input is empty."
    scalar = [f.name for f in frame.schema.fields if not f.dataType.simpleString().startswith("array")]
    nulls = frame.agg(*[F.sum(F.col(x).isNull().cast("long")).alias(x) for x in scalar]).first().asDict()
    b = frame.agg(F.min("game_ts").alias("first_game"), F.max("game_ts").alias("last_game"),
        F.min("white_elo").alias("min_white_elo"), F.max("white_elo").alias("max_white_elo"),
        F.avg("white_elo").alias("mean_white_elo"), F.avg("black_elo").alias("mean_black_elo")).first().asDict()
    return {"rows": n, "population": "all input games; no sample", "null_counts": nulls, **b,
        "n_plies": {"q25_q50_q75_q95": frame.approxQuantile("n_plies", [.25, .5, .75, .95], 0.01),
                    "quantile_rank_error": 0.01},
        "results": [r.asDict() for r in frame.groupBy("result").count().orderBy("result").collect()],
        "time_controls_top10": [r.asDict() for r in frame.groupBy("time_control").count().orderBy(F.desc("count")).limit(10).collect()],
        "days": [r.asDict() for r in frame.groupBy("utc_date").count().orderBy("utc_date").collect()]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Eligibility for the prediction
# MAGIC `eligibility_reason` is the Lichess counterpart of the Lab-6 metadata join. Silver keeps every **valid** game, but a game can only be used for the 20-ply prediction if it:
# MAGIC 1. was played in the file's month (`outside_month`: the monthly file also holds a few hours of the previous month),
# MAGIC 2. has a time control (`untimed_game`: correspondence games have none),
# MAGIC 3. reached 20 plies (`too_short`),
# MAGIC 4. has a clock value for each of the first 20 plies (`incomplete_clocks`).
# MAGIC 
# MAGIC A game that fails gets the **first** reason that applies, or `NULL` if it is eligible. These games are not deleted: validate saves them to quarantine with the reason.

# COMMAND ----------

def eligibility_reason(month):
    prefix = month.replace("-", ".")
    clocks_ok = F.coalesce((F.size("clocks20") == PLY_POINT) & F.forall("clocks20", lambda x: x.isNotNull()), F.lit(False))
    return (F.when(~F.coalesce(F.col("utc_date").startswith(prefix), F.lit(False)), "outside_month")
        .when(F.coalesce(F.col("time_control") == "-", F.lit(True)), "untimed_game")
        .when(F.coalesce(F.col("n_plies") < PLY_POINT, F.lit(True)), "too_short")
        .when(~clocks_ok, "incomplete_clocks"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. The quality gate
# MAGIC `evaluate_gate` checks the saved silver against named rules. Each rule has a **policy**:
# MAGIC - **fail**: if the rule is broken, the run stops and nothing is published.
# MAGIC - **quarantine**: games that break the rule are set aside with a reason; the run fails only if there are too many.
# MAGIC - **warn**: the problem is recorded, but the run continues.
# MAGIC 
# MAGIC | Rule | Policy | What it checks |
# MAGIC |---|---|---|
# MAGIC | `schema` | fail | Every column has the expected type. |
# MAGIC | `volume` | fail | The row count equals the count in your silver report. |
# MAGIC | `keys` | fail | No empty `game_id`. |
# MAGIC | `result` | fail | `result` is `1-0`, `0-1` or `1/2-1/2`. |
# MAGIC | `elo_range` | fail | Both Elo ratings are between 100 and 4000. |
# MAGIC | `timestamp` | fail | `game_ts` is present. |
# MAGIC | `time_control` | fail | Timed games have base >= 0, increment >= 0 and base + increment > 0 (Lichess allows `0+1`); untimed games (`-`) have no base. |
# MAGIC | `plies_arrays` | fail | `moves20` has `min(n_plies, 20)` entries and `clocks20` the same number. |
# MAGIC | `unique_game` | fail | No two rows share a `game_id`. |
# MAGIC | `eligibility` | quarantine | At most 15% of games are ineligible for the 20-ply prediction. |
# MAGIC | `clock_gap` | warn | Among otherwise-eligible games, at most 10% lack a clock value. |
# MAGIC 
# MAGIC A missing value can never pass a rule: each check uses `coalesce(..., False)`. The function returns the evidence and the verdict. The validate task saves `gate.json` first and only then raises an error, so a failed run still leaves its evidence.

# COMMAND ----------

def evaluate_gate(silver, expected_rows, month, tolerance=ELIGIBILITY_TOLERANCE):
    """Return evidence and a verdict. The task saves evidence before raising."""
    errors = schema_errors(silver)
    if errors:
        return {"passed": False, "failed_rules": ["schema"], "schema_errors": errors}, None
    n = silver.count()
    results = []
    def record(name, policy, passed, detail):
        results.append({"rule": name, "policy": policy, "passed": bool(passed), "detail": detail})
    record("schema", "fail", True, "silver column types match")
    record("volume", "fail", n == expected_rows and n > 0, {"actual": n, "expected": expected_rows})
    elo_ok = F.col("white_elo").between(ELO_MIN, ELO_MAX) & F.col("black_elo").between(ELO_MIN, ELO_MAX)
    tc_ok = F.when(F.col("time_control") == "-", F.col("tc_base").isNull()).otherwise(
        (F.col("tc_base") >= 0) & (F.col("tc_inc") >= 0) & (F.col("tc_base") + F.col("tc_inc") > 0))
    arrays_ok = (F.size("moves20") == F.least(F.col("n_plies"), F.lit(PLY_POINT))) & (F.size("clocks20") == F.size("moves20"))
    predicates = {"keys": F.length(F.trim("game_id")) > 0,
                  "result": F.col("result").isin(*RESULTS), "elo_range": elo_ok,
                  "timestamp": F.col("game_ts").isNotNull(), "time_control": tc_ok, "plies_arrays": arrays_ok}
    counts = silver.agg(*[F.sum((~F.coalesce(good, F.lit(False))).cast("long")).alias(name)
                          for name, good in predicates.items()]).first().asDict()
    for name, count in counts.items(): record(name, "fail", (count or 0) == 0, {"invalid_rows": count or 0})
    duplicates = silver.groupBy("game_id").count().filter("count > 1").count()
    record("unique_game", "fail", duplicates == 0, {"duplicate_ids": duplicates})
    ineligible = silver.withColumn("reject_reason", eligibility_reason(month)).filter("reject_reason IS NOT NULL")
    by = {r["reject_reason"]: r["count"] for r in ineligible.groupBy("reject_reason").count().collect()}
    quarantined = sum(by.values())
    record("eligibility", "quarantine", quarantined / max(n, 1) <= tolerance,
           {"quarantined_rows": quarantined, "fraction": quarantined / max(n, 1), "tolerance": tolerance, "by_reason": by})
    candidates = n - by.get("outside_month", 0) - by.get("untimed_game", 0) - by.get("too_short", 0)
    gap = by.get("incomplete_clocks", 0) / max(candidates, 1)
    record("clock_gap", "warn", gap <= CLOCK_GAP_WARN, {"incomplete_clocks": by.get("incomplete_clocks", 0), "fraction": gap})
    failed = [r["rule"] for r in results if not r["passed"] and r["policy"] != "warn"]
    return {"passed": not failed, "failed_rules": failed, "rules": results,
            "rows_in": n, "quarantined_rows": quarantined,
            "accepted_rows": n - quarantined, "eligibility_tolerance": tolerance}, ineligible

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Coverage: who is lost between silver and gold?
# MAGIC `coverage` compares the number of games per group (for example per result or per time-control class) before (silver) and after (gold), with `retained_pct` for each group. A good overall total can hide one group that lost many games. `speed_class_expr` is the time-control class: 0 bullet, 1 blitz, 2 rapid, 3 classical, from `base + 40 x increment` seconds.

# COMMAND ----------

def speed_class_expr():
    base = F.col("tc_base") + 40 * F.col("tc_inc")
    return (F.when(base.isNull(), F.lit(None).cast("int")).when(base < 180, 0).when(base < 480, 1)
            .when(base < 1500, 2).otherwise(3).cast("int"))

def coverage(silver, gold, column):
    """Compare full-population groups. Missing after-groups get zero rows."""
    before = silver.groupBy(column).agg(F.count("*").alias("before"))
    after = gold.groupBy(column).agg(F.count("*").alias("after"))
    return (before.join(after, column, "left").fillna(0, ["after"])
            .withColumn("retained_pct", F.round(100.0 * F.col("after") / F.col("before"), 3)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Features
# MAGIC `make_features` builds the columns described in `FEATURES.md` from **gold** (eligible silver games plus the `label`: 0 White win, 1 draw, 2 Black win). Only information known by ply 20 is used:
# MAGIC - Clocks are seconds left after each ply; index 0 is White's first move.
# MAGIC - `w_used_10` / `b_used_10` are seconds used over each player's first 10 moves.
# MAGIC - Result, rating changes, termination and ply count are **never** features (known only after the game).

# COMMAND ----------

def make_features(gold):
    k = F.col("clocks20")
    used_w, used_b = k.getItem(0) - k.getItem(18), k.getItem(1) - k.getItem(19)
    return gold.select("game_id", "game_ts", "label", "white_elo", "black_elo",
        (F.col("white_elo") - F.col("black_elo")).alias("elo_diff"),
        ((F.col("white_elo") + F.col("black_elo")) / 2.0).alias("elo_mean"),
        "tc_base", "tc_inc", speed_class_expr().alias("speed_class"), "eco",
        F.substring("eco", 1, 1).alias("eco_family"),
        F.trim(F.split("opening", ":").getItem(0)).alias("opening_family"),
        k.getItem(0).alias("w_clk_1"), k.getItem(1).alias("b_clk_1"),
        used_w.alias("w_used_10"), used_b.alias("b_used_10"), (used_w - used_b).alias("used_diff_10"),
        (k.getItem(0) != k.getItem(1)).cast("int").alias("unequal_start"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Loaded
# MAGIC This last cell confirms that the definitions above are ready. When a task notebook runs `%run ./lichess_00_common`, this line appears in that task's output too.

# COMMAND ----------

print(f"lichess_00_common loaded: {len(KEY)}-column game key, {len(EXPECTED_TYPES)} contract columns, "
      f"{len(FEATURE_COLUMNS)} feature columns, eligibility tolerance {ELIGIBILITY_TOLERANCE:.1%}.")
