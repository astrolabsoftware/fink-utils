from fink_utils.sso.periods import estimate_synodic_period
import requests
import io
import pandas as pd

ssnamenr = 5209

r = requests.post("https://fink-portal.org/api/v1/sso", json={"n_or_d": ssnamenr, "withEphem": True, "output-format": "json"})
pdf = pd.read_json(io.BytesIO(r.content))

flavor = "SHG1G2"
period, chi2red = estimate_synodic_period(
    pdf=pdf, flavor=flavor, sb_method="fastnifty", Nterms_base=1, period_range=(0.05, 1.2)
)
