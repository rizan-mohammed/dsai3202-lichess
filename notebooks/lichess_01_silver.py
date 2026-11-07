# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1: Bronze to silver (Lichess)
# MAGIC Read the complete bronze Parquet, apply a declared contract, account for every input row, and save cleaned games as Parquet files with a Delta log. This is the Lichess version of Lab-5 `01_silver`.
# MAGIC 
# MAGIC It runs as the **first task of the job** (`silver`), and you can also run it by hand, one section at a time. Widgets: `lake`, `catalog`, `schema`, `month`. Bronze is read from the staged copy in `processed/lichess/bronze_staged/` (the parsed Parquet that also sits in `raw/lichess/bronze/`).
# MAGIC 
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Reads** | `processed/lichess/bronze_staged/month=<month>/` and its `_bronze-manifest.json`. |
# MAGIC | **Writes** | Rejected games to `processed/lichess_silver_rejects/run_id=<id>/`, silver to `processed/lichess_silver` (Delta table `lichess_silver`), and the report `curated/quality/lichess_drop_report.json`. |

# COMMAND ----------

# 2.1 Run once to create widgets, enter your values, then rerun this cell.
dbutils.widgets.text("lake", "amazonlake60306369", "Existing Lab-3 lake")
dbutils.widgets.text("catalog", "dsai3202_databricks", "Instructor-assigned catalog")
dbutils.widgets.text("schema", "s_60306369", "Your student schema")
dbutils.widgets.text("month", "2017-05", "Lichess month")

import json
import re
import uuid
from datetime import datetime, timezone
from pyspark import StorageLevel
from pyspark.sql import functions as F, Window
from pyspark.sql.types import ArrayType

LAKE = dbutils.widgets.get("lake").strip()
CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
MONTH = dbutils.widgets.get("month").strip()
assert re.fullmatch(r"[a-z0-9]{3,24}", LAKE), "Enter the existing storage account name."
assert all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v) for v in [CATALOG, SCHEMA]), \
    "Enter your assigned catalog and schema (letters, digits and underscores)."
assert CATALOG != "hive_metastore", "This guide uses Unity Catalog. Ask for the assigned catalog."
assert re.fullmatch(r"\d{4}-\d{2}", MONTH), "month looks like 2017-05"

TABLE = f"{CATALOG}.{SCHEMA}.lichess_silver"
ROOT = f"abfss://processed@{LAKE}.dfs.core.windows.net"
SRC = f"{ROOT}/lichess/bronze_staged/month={MONTH}"
SILVER = f"{ROOT}/lichess_silver"
QUALITY = f"abfss://curated@{LAKE}.dfs.core.windows.net/quality"
RUN_ID = str(uuid.uuid4())
REJECTS = f"{ROOT}/lichess_silver_rejects/run_id={RUN_ID}"
ELO_MIN, ELO_MAX = 100, 4000
RESULTS = ["1-0", "0-1", "1/2-1/2"]
KEY = ["game_id"]
SILVER_COLUMNS = ["game_id", "event", "result", "utc_date", "game_ts", "white_elo", "black_elo",
    "white_diff", "black_diff", "eco", "opening", "time_control", "tc_base", "tc_inc",
    "termination", "n_plies", "moves20", "clocks20"]
STEPS = ["invalid_game_id", "invalid_result", "invalid_elo", "invalid_timestamp",
         "invalid_time_control", "duplicate"]
spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.shuffle.partitions", "400")   # smaller sort partitions: games carry two arrays
print("Month:", MONTH, "| bronze:", SRC, "| silver table:", TABLE)

# COMMAND ----------

# 2.2 Read the complete bronze output and check it against the parser manifest.
bronze_manifest = json.loads(dbutils.fs.head(f"{SRC}/_bronze-manifest.json", 65536))
try:
    source_manifest = json.loads(dbutils.fs.head(f"{SRC}/_source-manifest.json", 65536))
except Exception:
    source_manifest = {}
bronze = (spark.read.parquet(SRC)
          .select("*", F.col("_metadata.file_path").alias("_source_file")))
required = {"game_id", "event", "result", "utc_date", "utc_time", "white_elo", "black_elo", "white_diff",
            "black_diff", "eco", "opening", "time_control", "termination", "n_plies", "moves20", "clocks20"}
assert required <= set(bronze.columns), f"Missing bronze columns: {required - set(bronze.columns)}"
assert isinstance(bronze.schema["moves20"].dataType, ArrayType), "moves20 must stay an array."
bronze.printSchema()
rows_in = bronze.count()
print("rows_in =", rows_in, "| parser manifest games_parsed =", bronze_manifest["games_parsed"])
assert rows_in == bronze_manifest["games_parsed"], \
    "Bronze must contain every parsed game. Check missing or repeated files."
display(bronze.groupBy("utc_date").count().orderBy("utc_date"))
display(bronze.select(*sorted(required)).limit(10))
# Checkpoint: record schema, total rows and the first and last game days.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The cleaning contract
# MAGIC The first failed rule supplies a game's reason, checked in this order:
# MAGIC 1. `invalid_game_id`: empty id.
# MAGIC 2. `invalid_result`: not `1-0`, `0-1` or `1/2-1/2` (for example `*`, unfinished).
# MAGIC 3. `invalid_elo`: a rating is missing (anonymous or unrated players) or outside 100-4000.
# MAGIC 4. `invalid_timestamp`: the date and time cannot be read.
# MAGIC 5. `invalid_time_control`: not `base+increment` seconds and not `-`.
# MAGIC 
# MAGIC Among the valid games, a repeated `game_id` is a **duplicate**: keep the game with the most plies, then break ties on a hash of the complete record. This is reproducible, not a claim that the kept copy is the true one.
# MAGIC 
# MAGIC Games that are valid but cannot be used for the 20-ply prediction (short games, no time control, other month, missing clocks) are **not** rejected here: silver keeps them. Validate and publish handle them as *eligibility*.

# COMMAND ----------

# 3.1 Transformation function. This cell defines the rules without saving data.
def clean_games(frame):
    original = sorted(c for c in frame.columns if not c.startswith("_"))
    tc = F.col("time_control")
    timed = F.coalesce(tc.rlike(r"^\d+\+\d+$"), F.lit(False))
    untimed = F.coalesce(tc == "-", F.lit(False))
    typed = (frame
        .withColumn("tc_base", F.when(timed, F.expr("try_cast(split(time_control, '[+]')[0] AS int)")))
        .withColumn("tc_inc", F.when(timed, F.expr("try_cast(split(time_control, '[+]')[1] AS int)")))
        .withColumn("game_ts", F.expr("try_to_timestamp(concat(utc_date, ' ', utc_time), 'yyyy.MM.dd HH:mm:ss')")))
    elo_ok = F.col("white_elo").between(ELO_MIN, ELO_MAX) & F.col("black_elo").between(ELO_MIN, ELO_MAX)
    first_failed = (
        F.when(F.coalesce(F.col("game_id").isNull() | (F.col("game_id") == ""), F.lit(True)), "invalid_game_id")
        .when(~F.coalesce(F.col("result").isin(*RESULTS), F.lit(False)), "invalid_result")
        .when(~F.coalesce(elo_ok, F.lit(False)), "invalid_elo")
        .when(F.col("game_ts").isNull(), "invalid_timestamp")
        .when(~(timed | untimed), "invalid_time_control"))
    typed = typed.withColumn("_reason", first_failed).withColumn("_valid", F.col("_reason").isNull())
    # Duplicates are decided among valid games only.
    order = Window.partitionBy("game_id", "_valid").orderBy(F.col("n_plies").desc(), F.xxhash64(*original))
    typed = typed.withColumn("_rn", F.row_number().over(order))
    typed = typed.withColumn("reject_reason", F.coalesce(F.col("_reason"),
                             F.when(F.col("_rn") > 1, F.lit("duplicate"))))
    typed = typed.persist(StorageLevel.DISK_ONLY)   # disk only: the cache must not compete with the sort for executor memory
    silver = typed.filter("reject_reason IS NULL").select(*SILVER_COLUMNS)
    rejected = (typed.filter("reject_reason IS NOT NULL")
        .select("reject_reason", "_source_file", F.to_json(F.struct(*[F.col(c) for c in original])).alias("_raw_json")))
    return silver, rejected

# COMMAND ----------

# 3.2 Apply the rules and count every exit.
silver_out, rejected = clean_games(bronze)
rows_out = silver_out.count()
counts = {r["reject_reason"]: r["count"] for r in rejected.groupBy("reject_reason").count().collect()}
steps = {k: counts.get(k, 0) for k in STEPS}
n_dupes = steps["duplicate"]
n_rejected = sum(steps.values()) - n_dupes
assert rows_in == rows_out + n_rejected + n_dupes, "Row accounting does not balance."
assert rows_out > 0, "No games passed. Investigate the input and contract before publishing."
remaining = rows_in
for name, dropped in steps.items():
    remaining -= dropped
    print(f"{name:22s}: dropped {dropped:>9,}; remaining {remaining:>10,}")
print(f"BALANCED: {rows_in:,} = {rows_out:,} kept + {n_rejected:,} invalid + {n_dupes:,} duplicates")
print(f"Dropped: {100 * (rows_in - rows_out) / rows_in:.3f}%")
display(rejected.select("reject_reason", "_source_file", "_raw_json").limit(10))
# Checkpoint: record actual counts, rule order and a reason example.

# COMMAND ----------

# 4.1 Save both invalid games and duplicate copies, using one folder per run.
if n_rejected + n_dupes:
    rejected.coalesce(4).write.mode("overwrite").parquet(REJECTS)
    saved_rejects = spark.read.parquet(REJECTS)
    assert saved_rejects.count() == n_rejected + n_dupes
    display(saved_rejects.groupBy("reject_reason").count().orderBy("reject_reason"))
    print("Reject evidence:", REJECTS)
else:
    print("No rejected games; nothing to save.")
# Checkpoint: read the saved files, not just the in-memory frame.

# COMMAND ----------

# 5.1 Save silver data as Parquet + Delta log at a fixed lake location.
# CREATE TABLE registers its SQL name and columns. USING DELTA manages files and versions.
expected_schema = [(f.name, f.dataType) for f in silver_out.select(*SILVER_COLUMNS).schema]
ddl = ", ".join(f"{n} {t.simpleString().upper()}" for n, t in expected_schema)
spark.sql(f"CREATE TABLE IF NOT EXISTS {TABLE} ({ddl}) USING DELTA LOCATION '{SILVER}'")
detail = spark.sql(f"DESCRIBE DETAIL {TABLE}").first()
assert detail["format"] == "delta" and detail["location"].rstrip("/") == SILVER, \
    "The table name points somewhere else. Stop and ask the instructor."
assert [(f.name, f.dataType) for f in spark.table(TABLE).schema] == expected_schema, \
    "The existing table has different column names, order or types."
# Full rebuild. insertInto resolves by position, so column order matters.
silver_out.select(*SILVER_COLUMNS).coalesce(16).write.mode("overwrite").insertInto(TABLE)
first_version = spark.sql(f"DESCRIBE HISTORY {TABLE}").first()["version"]
assert spark.table(TABLE).count() == rows_out
display(spark.sql(f"DESCRIBE DETAIL {TABLE}").select("format", "location", "numFiles"))
display(spark.table(TABLE).groupBy("result").count().orderBy("result"))
# Checkpoint: location, version and rows by result.

# COMMAND ----------

# 5.2 Repeat the write, then compare all values. The fingerprint hashes every column of every row.
def fingerprint(frame):
    cols = sorted(frame.columns)
    h1 = F.pmod(F.xxhash64(*cols), F.lit(2147483647)).cast("long")
    h2 = F.pmod(F.hash(*cols[::-1]), F.lit(2147483647)).cast("long")
    return tuple(frame.select(*cols).agg(F.count(F.lit(1)), F.sum(h1), F.sum(h2)).first())

silver_out.select(*SILVER_COLUMNS).coalesce(16).write.mode("overwrite").insertInto(TABLE)
repeat_version = spark.sql(f"DESCRIBE HISTORY {TABLE}").first()["version"]
before = (spark.read.format("delta").option("versionAsOf", first_version).load(SILVER).select(*SILVER_COLUMNS))
after = spark.table(TABLE).select(*SILVER_COLUMNS)
assert fingerprint(before) == fingerprint(after), "The repeated write changed values."
print(f"SAME CONTENT: versions {first_version} and {repeat_version}; {rows_out:,} rows")
display(spark.sql(f"DESCRIBE HISTORY {TABLE}").select("version", "timestamp", "operation").limit(5))
# Checkpoint: location, version numbers and equal-content result.
# New versions are expected. A repeatable run preserves the current game values.

# COMMAND ----------

# 7.1 Save completion evidence last, recording this run's files and silver data version.
dropped = rows_in - rows_out
report = {
    "lab": "phase1", "run_id": RUN_ID, "month": MONTH, "completed_utc": datetime.now(timezone.utc).isoformat(),
    "source": SRC, "table": TABLE, "silver_path": SILVER, "silver_version": repeat_version,
    "rejects_path": REJECTS if (n_rejected + n_dupes) else None, "rows_in": rows_in, "rows_out": rows_out,
    "invalid_rows": n_rejected, "duplicate_rows": n_dupes,
    "dropped": dropped, "dropped_pct": round(100 * dropped / rows_in, 3),
    "dropped_by_step": steps, "bronze_manifest": bronze_manifest,
    "source_sha256": source_manifest.get("sha256"),
    "harshest_filter": max(steps, key=steps.get) if dropped else "none",
    "rerun_same_content": True,
}
text = json.dumps(report, indent=2)
dbutils.fs.mkdirs(f"{QUALITY}/lichess_silver_runs")
dbutils.fs.put(f"{QUALITY}/lichess_silver_runs/{RUN_ID}.json", text, overwrite=False)
dbutils.fs.put(f"{QUALITY}/lichess_drop_report.json", text, overwrite=True)
saved = json.loads(dbutils.fs.head(f"{QUALITY}/lichess_drop_report.json", 65536))
assert saved == report
assert spark.table(TABLE).count() == rows_out
print(text)
spark.catalog.clearCache()
# Checkpoint: save the JSON and export the notebook with displayed results.
