# Test data generation

## colnames_ztf.csv

```python
from fink_utils.spark.utils import return_flatten_names

df = ...

fl = return_flatten_names(df, flatten_schema=[])
pd.DataFrame({"colnames": [fl]}).to_csv("colnames.csv")
```
