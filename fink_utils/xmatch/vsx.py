# Copyright 2019-2026 AstroLab Software
# Author: Anais Möller
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def return_list_of_nonstellar() -> list:
    """Non-stellar variable objects (AGN, quasars, supernovae, etc.)

    AAVSO Variable Star Index (VSX) Classification Lists
    Source: https://vsx.aavso.org/index.php?view=about.vartypessort

    Returns
    -------
    out: list
        List of labels
    """
    vsx_notstars = [
        "AGN",  # Active Galactic Nuclei
        "BLLAC",  # BL Lacertae-type objects (extragalactic)
        "QSO",  # Quasi-stellar objects (quasars)
        "GRB",  # Gamma Ray Burst optical counterparts
        "Microlens",  # Microlensing events
        "TDE",  # Tidal Disruption Events
        "LFBOT",  # Luminous Fast Blue Optical Transients
        "SN",  # Supernovae (general)
        "SN I",  # Type I supernovae
        "SN Ia",  # Type Ia supernovae
        "SN Ia-00cx-like",
        "SN Ia-02es-like",
        "SN Ia-06gz-like",
        "SN Ia-86G-like",
        "SN Ia-91bg-like",
        "SN Ia-91T-like",
        "SN Ia-99aa-like",
        "SN Ia-Ca-rich",
        "SN Ia-CSM",
        "SN Iax",
        "SN Ib",
        "SN Ic",
        "SN Ic-BL",  # Broad-lined SN Ic
        "SN Icn",
        "SN Idn",
        "SN Ien",
        "SN II",  # Type II supernovae
        "SN II-L",
        "SN II-P",
        "SN IIa",
        "SN IIb",
        "SN IId",
        "SN IIn",
        "SN-pec",  # Peculiar supernovae
        "SLSN",  # Super Luminous Supernovae
        "SLSN-I",
        "SLSN-II",
    ]
    return vsx_notstars


def return_list_of_stellar() -> list:
    """Stellar variable objects (e.g. regular, irregular, eclipsing variable stars)

    AAVSO Variable Star Index (VSX) Classification Lists
    Source: https://vsx.aavso.org/index.php?view=about.vartypessort

    Returns
    -------
    out: list
        List of labels
    """
    # Variable stars (actual stellar objects)
    vsx_variable_stars = [
        "*",  # Unique variable stars outside normal classifications
        "ACEP",  # Anomalous Cepheids (fundamental mode)
        "ACEP(B)",  # Anomalous Cepheids (multi-mode)
        "ACEPS",  # Anomalous Cepheids (first overtone)
        "CEP",  # Cepheids (general)
        "DCEP",  # Delta Cephei-type (classical Cepheids)
        "DCEP(B)",  # Classical Cepheids (multi-mode)
        "DCEP-FO",  # First Overtone classical Cepheids
        "DCEP-FU",  # Fundamental mode classical Cepheids
        "DCEPS",  # Delta Cep variables (first overtone)
        "DCEPS(B)",  # First/second overtone double-mode Cepheids
        "CW",  # W Virginis variables
        "CW-FO",  # First Overtone CW stars
        "CW-FU",  # Fundamental mode CW stars
        "CWA",  # W Virginis variables (long period)
        "CWB",  # BL Herculis variables (short period W Vir)
        "CWB(B)",  # BL Herculis (multi-mode)
        "CWBS",  # BL Herculis (first overtone)
        "RR",  # RR Lyrae variables (general)
        "RRAB",  # RR Lyrae (asymmetric, fundamental mode)
        "RRC",  # RR Lyrae (symmetric, overtone)
        "RRD",  # RR Lyrae (double-mode)
        "RV",  # RV Tauri variables
        "RVA",  # RV Tauri (constant mean magnitude)
        "RVB",  # RV Tauri (variable mean magnitude)
        "DSCT",  # Delta Scuti variables
        "DSCTC",  # Low-amplitude Delta Scuti (obsolete)
        "DSCTr",  # Delta Scuti subtype
        "HADS",  # High Amplitude Delta Scuti
        "HADS(B)",  # HADS (multi-mode)
        "SXPHE",  # SX Phoenicis variables
        "SXPHE(B)",  # SX Phe (multi-mode)
        "GDOR",  # Gamma Doradus stars
        "BCEP",  # Beta Cephei variables
        "BCEPS",  # Beta Cep (short period) - obsolete
        "SPB",  # Slowly Pulsating B stars
        "SPBe",  # SPBe stars (rapidly rotating)
        "ACYG",  # Alpha Cygni variables
        "M",  # Mira variables
        "SR",  # Semi-regular variables
        "SRA",  # Semi-regular (persistent periodicity)
        "SRB",  # Semi-regular (poorly defined periodicity)
        "SRC",  # Semi-regular supergiants
        "SRD",  # Semi-regular F-K type
        "SRS",  # Semi-regular (short period)
        "L",  # Slow irregular variables
        "LB",  # Slow irregular (late type)
        "LC",  # Irregular supergiants (late type)
        "I",  # Poorly studied irregular variables
        "IA",  # Irregular (early spectral type)
        "IB",  # Irregular (intermediate to late type)
        "IS",  # Rapid irregular variables
        "ISA",  # Rapid irregular (early type)
        "ISB",  # Rapid irregular (intermediate/late type)
        "IN",  # Orion variables
        "INA",  # Orion variables (early type)
        "INAT",  # INT-type with abrupt fadings
        "INB",  # Orion variables (intermediate/late type)
        "INS",  # IN stars (rapid variations)
        "INSA",  # ISA stars in nebulosity
        "INSB",  # ISB stars in nebulosity
        "INST",  # INT stars (rapid variations)
        "INT",  # T Tauri type in diffuse nebulae
        "TTS",  # T Tauri Stars
        "TTS/ROT",  # T Tauri with rotational variability
        "CTTS",  # Classical T Tauri Stars
        "CTTS/ROT",  # Classical T Tauri with rotational variability
        "WTTS",  # Weak-lined T Tauri Stars
        "WTTS/ROT",  # Weak-lined T Tauri with rotational variability
        "EXOR",  # EX Lupi type (EXors)
        "FUOR",  # FU Orionis type (FUors)
        "GCAS",  # Gamma Cassiopeiae type
        "BE",  # Be stars (variable, no outbursts)
        "SDOR",  # S Doradus variables (LBV)
        "UV",  # UV Ceti type (flare stars)
        "UVN",  # Flaring Orion variables
        "WR",  # Wolf-Rayet variables
        "FF",  # Final Flash objects
        "V838MON",  # V838 Mon type (Luminous Red Novae)
        "N",  # Novae (general)
        "NA",  # Fast novae
        "NB",  # Slow novae
        "NC",  # Very slow novae
        "NR",  # Recurrent novae
        "NL",  # Nova-like variables
        "NL/VY",  # Anti-dwarf novae (VY Scl)
        "UG",  # U Geminorum type (dwarf novae)
        "UGSS",  # SS Cygni type
        "UGSU",  # SU UMa type
        "UGER",  # ER UMa type
        "UGWZ",  # WZ Sge type
        "UGZ",  # Z Cam type
        "UGZ/IW",  # IW And stars (anomalous Z Cam)
        "ZAND",  # Z Andromedae type
        "CV",  # Cataclysmic variables (unspecified)
        "AM",  # AM Her type (polars)
        "DQ",  # DQ Her type (intermediate polars)
        "DQ/AE",  # Propellers (AE Aqr subtype)
        "CBSS",  # Close-binary supersoft source
        "CBSS/V",  # V Sge type
        "IBWD",  # Interacting Binary White Dwarfs
        "E",  # Eclipsing binaries (general)
        "E-DO",  # Disk occultation systems
        "EA",  # Beta Persei type (Algol)
        "EB",  # Beta Lyrae type
        "EC",  # Contact binaries
        "ED",  # Detached eclipsing binaries
        "ELL",  # Rotating ellipsoidal variables
        "EP",  # Planetary transits
        "ESD",  # Semi-detached eclipsing binaries
        "EW",  # W UMa type
        "DPV",  # Double Periodic Variables
        "HB",  # Heartbeat stars
        "R",  # Close binaries with reflection effect
        "RS",  # RS Canum Venaticorum type
        "BY",  # BY Draconis type
        "FKCOM",  # FK Comae Berenices type
        "ACV",  # Alpha2 CVn variables
        "SXARI",  # SX Arietis type
        "SXARI/E",  # Sigma Ori E stars
        "LERI",  # Lambda Eri type
        "roAp",  # Rapidly oscillating Ap stars
        "roAm",  # Rapidly oscillating Am stars
        "ROT",  # Spotted stars (general)
        "X",  # X-ray sources (general)
        "HMXB",  # High Mass X-ray Binaries
        "IMXB",  # Intermediate-mass X-ray Binaries
        "LMXB",  # Low Mass X-ray Binaries
        "PSR",  # Optically variable pulsars
        "WDP",  # White dwarf pulsars
        "ZZ",  # ZZ Ceti variables
        "ZZ/GWLIB",  # GW Lib stars
        "ZZA",  # ZZ Cet (DA type)
        "ZZA/O",  # ZZA with outbursts
        "ZZB",  # V777 Her type (DB)
        "ZZLep",  # ZZ Lep (PN central stars)
        "ZZO",  # GW Vir type (DO)
        "V361HYA",  # EC 14026 variables (rapid)
        "V1093HER",  # PG 1716 variables (slow)
        "DWLYN",  # Hybrid subdwarf pulsators
        "BLAP",  # Blue Large Amplitude Pulsators
        "FSCMa",  # FS CMa type
        "cPNB[e]",  # Compact PN B[e] stars
        "RCB",  # R CrB variables
        "DYPer",  # DY Per type
        "BXCIR",  # BX Cir type
        "PVTEL",  # PV Tel type
        "PVTELI",  # PV Tel type I
        "PVTELII",  # PV Tel type II
        "PVTELIII",  # PV Tel type III
        "YHG",  # Yellow hypergiants
        "PPN",  # Proto-planetary nebulae
        "ORG",  # Oscillating Red Giants
        "UXOR",  # UX Orionis stars
        "VBD",  # Variable brown dwarfs
        "APER",  # Aperiodic
        "CST",  # Non-variable (constant)
        "MISC",  # Miscellaneous variables
        "LPV",  # Long Period Variables
        "NSIN",  # Non-sinusoidal periodic
        "NSIN ELL",  # Non-sinusoidal ellipsoidal
        "PER",  # Periodic (unspecified, survey type)
        "PULS",  # Pulsating (unspecified, survey type)
        "S",  # Rapid light changes (unstudied)
        "SIN",  # Sinusoidal
        "Transient",  # UV transient
        "VAR",  # Variable (unspecified, survey type)
        "YSO",  # Young Stellar Object
        "non-cv",  # Non-cataclysmic
    ]
    return vsx_variable_stars
