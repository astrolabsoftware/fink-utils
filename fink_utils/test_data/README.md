# Test data generation

## colnames_ztf.csv

```python
from fink_utils.spark.utils import return_flatten_names

df = ...

fl = return_flatten_names(df, flatten_schema=[])
pd.DataFrame({"colnames": [fl]}).to_csv("colnames.csv")
```

## Rubin SSOFT data

Connect to the Rubin cluster, and execute:

```python
import pyspark.sql.functions as F

# Take one month of data
df = (
    spark.read
    .format("parquet")
    .option("mergeSchema", "true")
    .load("archive/science/year=2026/month=06")
)

# Check long object over a month
df.filter(df["mpc_orbits"].isNotNull()).groupBy(
    "mpc_orbits.unpacked_primary_provisional_designation"
).count().orderBy("count", ascending=False).show()

# Take one
sub = df.filter(
    df["mpc_orbits.unpacked_primary_provisional_designation"] == "2000 CL27"
).cache()

# Check
sub.count()
# Out[7]: 25

# Add partitioning columns (day is already there)
sub = sub.withColumn("month", F.lit("06"))
sub = sub.withColumn("year", F.lit("2026"))

# Write on HDFS
sub.write.partitionBy("year", "month", "day").parquet("rubin_sso_test_data")
```

Then transfer the data:

```bash
# on the cluster
hdfs dfs -get rubin_sso_test_data .

# on the local machine
scp -r fink@IP:rubin_sso_test_data fink_utils/test_data/
```
