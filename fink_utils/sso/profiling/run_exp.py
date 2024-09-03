from fink_utils.sso.periods import estimate_synodic_period, extract_physical_parameters
import requests
import io
import pandas as pd

ssnamenr = 5209
flavor = "SHG1G2"

r = requests.post(
    "https://fink-portal.org/api/v1/sso",
    json={"n_or_d": ssnamenr, "withEphem": True, "output-format": "json"},
)
pdf = pd.read_json(io.BytesIO(r.content))

phyparam = extract_physical_parameters(pdf, flavor)

period, chi2red = estimate_synodic_period(
    pdf=pdf,
    flavor=flavor,
    phyparam=phyparam,
    sb_method="fastnifty",
    Nterms_base=1,
    period_range=(1.0 / 24, 1.2),
)
print(period)
