from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys

YM = sys.argv[1] if len(sys.argv) > 1 else "2019-01"

year, month = YM.split("-")
next_month  = int(month) + 1
next_year   = int(year)
if next_month > 12:
    next_month = 1
    next_year += 1

start_date = f"{year}-{month}-01"
end_date   = f"{next_year}-{next_month:02d}-01"
INPUT      = f"data/raw/yellow_tripdata_{YM}.parquet"
OUTPUT     = f"data/processed/cleaned_{YM}.parquet"

spark = SparkSession.builder \
    .appName(f"clean-{YM}") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(INPUT)
print(f"{YM} raw rows: {df.count()}")

# Drop unneeded columns
for col in ["airport_fee", "store_and_fwd_flag"]:
    if col in df.columns:
        df = df.drop(col)

# Filter to valid dates
df = df.filter(
    (F.col("tpep_pickup_datetime") >= start_date) &
    (F.col("tpep_pickup_datetime") <  end_date)
)

# Fill and drop nulls
df = df.fillna({"congestion_surcharge": 0.0})
df = df.dropna(subset=["passenger_count", "RatecodeID", "PULocationID"])

print(f"{YM} clean rows: {df.count()}")
df.write.mode("overwrite").parquet(OUTPUT)
print(f"Saved to {OUTPUT}")

spark.stop()