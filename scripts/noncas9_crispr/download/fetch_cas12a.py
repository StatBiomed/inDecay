"""
Download Cas12a data from Zhu et al. 2021.

Source: DOI 10.3390/ijms222413301
SRA BioProject: PRJNA773803 (raw FASTQ fallback)
Preferred: Supplementary processed data from the MDPI article.
"""

import logging
from pathlib import Path

import requests

from scripts.noncas9_crispr.config import CAS12A_RAW_DIR, CAS12A_DOI

logger = logging.getLogger(__name__)

# MDPI supplementary data URLs for Zhu et al. 2021
# The paper includes supplementary tables with processed indel outcomes
MDPI_BASE = "https://www.mdpi.com/article/10.3390/ijms222413301"
SUPPLEMENTARY_URLS = {
    "supplementary_data": f"{MDPI_BASE}/s1",
}


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress logging."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        logger.info(f"Already downloaded: {dest_path.name}")
        return True

    logger.info(f"Downloading from {url}...")
    try:
        resp = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False

    with open(dest_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)

    logger.info(f"Downloaded: {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")
    return True


def fetch_cas12a_supplementary():
    """Download supplementary data from the Zhu et al. 2021 MDPI article.

    The supplementary materials contain processed indel outcome tables
    in [DEL_start_pos, DEL_len]:read_counts format.

    Returns:
        Path to the downloaded data directory.
    """
    CAS12A_RAW_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_any = False
    for name, url in SUPPLEMENTARY_URLS.items():
        dest = CAS12A_RAW_DIR / f"{name}.zip"
        if download_file(url, dest):
            downloaded_any = True

    if not downloaded_any:
        logger.warning(
            "Could not download supplementary data automatically. "
            "Please download manually from the MDPI article page: "
            f"https://doi.org/{CAS12A_DOI}"
        )
        logger.info(
            "Place the supplementary data files (Excel/CSV with indel outcomes) in: "
            f"{CAS12A_RAW_DIR}"
        )
        logger.info(
            "Expected format: tables with columns for guide ID, "
            "[DEL_start_pos, DEL_len], and read counts."
        )

    return CAS12A_RAW_DIR


def check_cas12a_data(data_dir=None):
    """Check if Cas12a data is available and identify file format.

    Args:
        data_dir: Directory to check. Defaults to CAS12A_RAW_DIR.

    Returns:
        dict with keys: 'available' (bool), 'files' (list), 'format' (str).
    """
    data_dir = Path(data_dir) if data_dir else CAS12A_RAW_DIR

    if not data_dir.exists():
        return {'available': False, 'files': [], 'format': None}

    # Look for data files in common formats
    patterns = ['*.xlsx', '*.csv', '*.tsv', '*.txt', '*.zip']
    found_files = []
    for pattern in patterns:
        found_files.extend(data_dir.glob(pattern))
        found_files.extend(data_dir.rglob(pattern))

    # Deduplicate
    found_files = sorted(set(found_files))

    # Determine format
    fmt = None
    for f in found_files:
        if f.suffix in ('.xlsx', '.xls'):
            fmt = 'excel'
            break
        elif f.suffix in ('.csv', '.tsv', '.txt'):
            fmt = 'text'
            break

    return {
        'available': len(found_files) > 0,
        'files': [str(f) for f in found_files],
        'format': fmt,
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fetch_cas12a_supplementary()
