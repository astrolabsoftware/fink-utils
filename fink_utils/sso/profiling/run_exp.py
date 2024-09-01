from fink_utils.sso.periods import extract_period_from_number

ssnamenr = 5209
flavor = "SHG1G2"
period, chi2red = extract_period_from_number(ssnamenr, flavor, Nterms_base=1, period_range=(0.05, 1.2))
