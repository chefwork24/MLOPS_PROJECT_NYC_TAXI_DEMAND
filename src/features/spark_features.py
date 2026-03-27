from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import sys

YM     = sys.argv[1] if len(sys.argv) > 1 else "2019-01"
INPUT  = f"data/processed/cleaned_{YM}.parquet"
OUTPUT = f"data/features/features_{YM}.parquet"

spark = SparkSession.builder \
    .appName(f"features-{YM}") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(INPUT)

# Aggregate to zone-hour demand
df = df.withColumn("pickup_hour", F.date_trunc("hour", F.col("tpep_pickup_datetime")))
demand = df.groupBy("PULocationID", "pickup_hour") \
           .count() \
           .withColumnRenamed("count", "demand")

# Temporal features
demand = demand \
    .withColumn("hour_of_day",  F.hour("pickup_hour")) \
    .withColumn("day_of_week",  F.dayofweek("pickup_hour")) \
    .withColumn("month",        F.month("pickup_hour")) \
    .withColumn("is_weekend",   F.dayofweek("pickup_hour").isin([1,7]).cast("int")) \
    .withColumn("is_rush_hour", F.hour("pickup_hour").isin([7,8,9,17,18,19]).cast("int"))

# Lag features
zone_window = Window.partitionBy("PULocationID").orderBy("pickup_hour")
demand = demand \
    .withColumn("demand_lag_1h",   F.lag("demand", 1).over(zone_window)) \
    .withColumn("demand_lag_24h",  F.lag("demand", 24).over(zone_window)) \
    .withColumn("demand_lag_168h", F.lag("demand", 168).over(zone_window))

# Rolling averages
demand = demand \
    .withColumn("rolling_mean_24h", F.avg("demand").over(zone_window.rowsBetween(-24,  -1))) \
    .withColumn("rolling_mean_7d",  F.avg("demand").over(zone_window.rowsBetween(-168, -1)))

# Zone flags
demand = demand \
    .withColumn("is_airport_zone", F.col("PULocationID").isin([1, 132, 138]).cast("int")) \
    .withColumn("ym", F.lit(YM))

print(f"{YM} feature rows: {demand.count()}")
print(f"Columns: {demand.columns}")
demand.write.mode("overwrite").parquet(OUTPUT)
print(f"Saved to {OUTPUT}")

spark.stop()