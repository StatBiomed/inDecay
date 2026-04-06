"""
Download Cas9-TREX2 data from the FORECasT Figshare repository.

Source: 10.6084/m9.figshare.7312067 (processed mutational profiles)
Uses Figshare API to list and download files.
"""

import logging
import zipfile
from pathlib import Path

import requests

from scripts.noncas9_crispr.config import FIGSHARE_API_URL, FIGSHARE_RAW_DIR, TREX2_REFERENCE_URL, TREX2_REFERENCE_FILE

logger = logging.getLogger(__name__)


def list_figshare_files():
    """List all files in the Figshare article.

    Returns:
        List of dicts with keys: name, size, download_url.
    """
    logger.info(f"Querying Figshare API: {FIGSHARE_API_URL}")
    resp = requests.get(FIGSHARE_API_URL, timeout=30)
    resp.raise_for_status()
    files = resp.json()
    logger.info(f"Found {len(files)} files in Figshare article")
    for f in files:
        logger.info(f"  {f['name']} ({f['size'] / 1e6:.1f} MB)")
    return files


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress logging.

    Args:
        url: Download URL.
        dest_path: Local destination path.
        chunk_size: Download chunk size in bytes.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        logger.info(f"Already downloaded: {dest_path.name}")
        return

    logger.info(f"Downloading {dest_path.name}...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total_size = int(resp.headers.get('content-length', 0))
    downloaded = 0

    with open(dest_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0 and downloaded % (chunk_size * 1000) < chunk_size:
                pct = downloaded * 100 / total_size
                logger.info(f"  {pct:.0f}% ({downloaded / 1e6:.1f} MB)")

    logger.info(f"Downloaded: {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")


def extract_zip(zip_path, extract_to):
    """Extract a ZIP file.

    Args:
        zip_path: Path to the ZIP file.
        extract_to: Directory to extract into.
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    logger.info("Extraction complete")


def download_reference_file():
    """Download the TREX2 reference sequence file from Springer supplementary materials.

    Downloads Supplementary Table 19 from Allen et al. 2019 (Nature Biotech)
    containing oligo IDs, guide sequences, target sequences, PAM indices, and strand info.

    Returns:
        Path to the downloaded reference file.
    """
    FIGSHARE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = FIGSHARE_RAW_DIR / TREX2_REFERENCE_FILE
    download_file(TREX2_REFERENCE_URL, dest)
    return dest


def fetch_forecast_data():
    """Download and extract FORECasT processed mutational profiles from Figshare.

    Downloads the ZIP archive containing mapped indel summaries for all
    experimental conditions including TREX2, plus the reference sequence file.

    Returns:
        Path to the extracted data directory.
    """
    FIGSHARE_RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = list_figshare_files()

    # Download all files (or filter for the processed profiles ZIP)
    for file_info in files:
        dest = FIGSHARE_RAW_DIR / file_info['name']
        download_file(file_info['download_url'], dest)

        # Extract ZIP files
        if dest.suffix == '.zip':
            extract_dir = FIGSHARE_RAW_DIR / dest.stem
            if not extract_dir.exists():
                extract_zip(dest, extract_dir)

    # Download reference sequence file
    download_reference_file()

    logger.info(f"FORECasT data available at: {FIGSHARE_RAW_DIR}")
    return FIGSHARE_RAW_DIR


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fetch_forecast_data()
