"""Configuration constants and paths for non-Cas9 CRISPR data preprocessing."""

import os
from pathlib import Path

# scripts/preprocess/ is two levels below inDecay root
_PREPROCESS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PREPROCESS_DIR.parent.parent   # inDecay root

# Raw downloaded data lives inside scripts/preprocess/data/
_LOCAL_DATA_DIR = _PREPROCESS_DIR / "data"
DATA_DIR = _LOCAL_DATA_DIR
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Separate alias so callers can also reference the inDecay data root
INDECAY_DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw data subdirectories
FIGSHARE_RAW_DIR = RAW_DIR / "figshare"
CAS12A_RAW_DIR = RAW_DIR / "cas12a"

# Processed output directories
TREX2_OUTPUT_DIR = PROCESSED_DIR / "trex2"
CAS12A_OUTPUT_DIR = PROCESSED_DIR / "cas12a"

# SelfTarget reference — resolved from env var or inDecay PATH.py at runtime
SELFTARGET_DIR = Path(os.environ.get(
    "SELFTARGET_DIR",
    "/PATH/TO/SelfTarget/SelfTarget",  # TODO: set this path for your environment
))
SELFTARGET_PYUTILS = SELFTARGET_DIR / "selftarget_pyutils"

# Figshare download
FIGSHARE_ARTICLE_ID = "7312067"
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"
FIGSHARE_DOI = "10.6084/m9.figshare.7312067"

# TREX2 reference sequence file (Supplementary Table 19 from Allen et al. 2019, Nature Biotech)
TREX2_REFERENCE_URL = "https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.4317/MediaObjects/41587_2019_BFnbt4317_MOESM72_ESM.txt"
TREX2_REFERENCE_FILE = "reference.txt"

# Cas12a source (Zhu et al. 2021)
CAS12A_DOI = "10.3390/ijms222413301"
CAS12A_SRA_BIOPROJECT = "PRJNA773803"

# inDecay format parameters
INDECAY_SEQ_LEN = 60          # 60bp reference window
INDECAY_CUT_POSITION = 30     # Cut site at position 30 in the 60bp window
MIN_READS_THRESHOLD = 5     # Minimum total reads per OligoID

# Cas12a cut geometry (configurable)
CAS12A_NTS_CUT_FROM_PAM = 18  # Non-target strand cut ~18nt from PAM
CAS12A_TS_CUT_FROM_PAM = 23   # Target strand cut ~23nt from PAM
CAS12A_CUT_REF_FROM_PAM = 20  # Midpoint reference (configurable)
CAS12A_OLIGO_LEN = 142        # Total oligo length in Zhu et al.
CAS12A_TARGET_LEN = 42        # Target region within the oligo

# FASTA header format for inDecay: >OligoID PAM_LOC PAM_DIR
# cut_idx = PAM_LOC - 3  (for FORWARD PAM)
# To place cut at position 30: PAM_LOC = 33
INDECAY_PAM_LOC = 33
INDECAY_PAM_DIR = "FORWARD"

# Output file names
INDEL_PROFILES_CSV = "indel_profiles.csv"
REFERENCE_FASTA = "reference.fasta"

# SelfTarget binaries — override via env vars if running outside Singularity
INDELGENTARGET_BIN = os.environ.get(
    "INDELGENTARGET_BIN",
    str(PROJECT_ROOT / "tool" / "indelgentarget"),
)
INDELGEN_BIN = os.environ.get(
    "INDELGEN_BIN",
    str(PROJECT_ROOT / "tool" / "indelgen"),
)