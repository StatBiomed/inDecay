"""
Extract TREX2 sample directories from FORECasT Figshare download.

Identifies K562-TREX2 directories using the same logic as
parseSampleName in selftarget/data.py (lines 88-109).

Filters for:
- Directories with 'TREX2' in name (NOT '2A_TREX2')
- DPI7 samples only (line 163: trex_dpi7 selector)
"""

import logging
import os
import re
from pathlib import Path

from scripts.noncas9_crispr.config import FIGSHARE_RAW_DIR

logger = logging.getLogger(__name__)


def parse_sample_name(dirname):
    """Parse sample metadata from directory name.

    Reimplements parseSampleName from selftarget/data.py:88-109.

    Args:
        dirname: Directory name string.

    Returns:
        dict with keys: cellline, dpi, virus, cov, date, month, dirname.
    """
    a_dirname = dirname
    # Handle the special K562_800x_7A_DPI7_may case
    if 'K562_800x_7A_DPI7_may' in dirname:
        dirname = dirname[:-8] + 'DPI16_may'

    toks = dirname.split('_') if '_' in dirname else dirname.split('-')

    # Extract DPI
    dpi = 0
    for tok in toks:
        if 'DPI' in tok:
            try:
                dpi = int(tok[3:])
            except ValueError:
                pass
            break

    # Extract cell line
    cellline = ''
    for x in ['K562', 'RPE1', 'CAS9', 'eCAS9', 'WT', 'TREX2', '2A_TREX2',
              'HAP1', 'E14TG2A', 'CHO', 'BOB', 'SIM']:
        if x in dirname:
            cellline = x
    if cellline == 'CAS9':
        cellline = 'K562'
    if cellline == '2A_TREX2':
        cellline = 'TREX2_2A'
    if cellline == 'WT':
        cellline = ' WT'

    # Extract coverage
    cov = ''
    for x in ['1600x', '800x', '500x', '1600X']:
        if x in dirname:
            cov = x

    # Extract virus batch
    virus = ''
    for x in ['12NA', '12NB', '7A', '7B', '6OA', '6OB', '12OA', '12OB']:
        if x in dirname:
            virus = x

    return {
        'cellline': cellline,
        'dpi': dpi,
        'virus': virus,
        'cov': cov,
        'dirname': a_dirname,
    }


def is_old_lib(dirname):
    """Check if a directory name indicates the old library scaffold."""
    if ('O' in dirname and 'BOB' not in dirname and 'CHO' not in dirname) \
            or 'Old' in dirname or 'old' in dirname:
        return True
    return False


def is_trex2_dpi7(dirname):
    """Check if a directory is a TREX2 DPI7 sample.

    Reimplements the trex_dpi7 selector from data.py:163:
      trex_dpi7 = lambda x: parseSampleName(x)[0] == 'TREX2'
                             and parseSampleName(x)[1] == 7
                             and all_sel(x)

    Args:
        dirname: Directory name.

    Returns:
        True if this is a TREX2 DPI7 sample (not 2A_TREX2).
    """
    info = parse_sample_name(dirname)
    # Must be TREX2 (not TREX2_2A which comes from '2A_TREX2')
    if info['cellline'] != 'TREX2':
        return False
    if info['dpi'] != 7:
        return False
    # Apply good_sel filter (exclude known bad samples)
    if 'NULL' in dirname or 'WT' in dirname:
        return False
    # Exclude the specific bad 2A_TREX2 12NB DPI3 sample
    if '2A_TREX' in dirname and '12NB' in dirname and 'DPI3' in dirname:
        return False
    return True


def find_trex2_dirs(data_root=None):
    """Find all TREX2 DPI7 sample directories in the Figshare download.

    Args:
        data_root: Root directory to search. Defaults to FIGSHARE_RAW_DIR.

    Returns:
        List of Path objects pointing to TREX2 DPI7 sample directories.
    """
    data_root = Path(data_root) if data_root else FIGSHARE_RAW_DIR
    trex2_dirs = []

    if not data_root.exists():
        logger.warning(f"Data root does not exist: {data_root}")
        return trex2_dirs

    # Search for directories matching TREX2 pattern
    for root, dirs, files in os.walk(data_root):
        for d in dirs:
            full_path = Path(root) / d
            if is_trex2_dpi7(d):
                trex2_dirs.append(full_path)
                logger.info(f"Found TREX2 DPI7 directory: {d}")

    logger.info(f"Found {len(trex2_dirs)} TREX2 DPI7 directories")
    return sorted(trex2_dirs)


def find_wt_dir(trex2_dir):
    """Find the corresponding wild-type directory for a TREX2 sample.

    WT samples are needed for background subtraction during profile reading.

    Args:
        trex2_dir: Path to a TREX2 sample directory.

    Returns:
        Path to the WT directory, or None if not found.
    """
    # WT directories follow the pattern: WT_12NA_DPI7 or WT_12OA_DPI7
    dirname = trex2_dir.name
    if is_old_lib(dirname):
        wt_pattern = 'WT_12OA_DPI7'
    else:
        wt_pattern = 'WT_12NA_DPI7'

    # Search in same parent structure
    for root, dirs, files in os.walk(trex2_dir.parent.parent):
        for d in dirs:
            if wt_pattern in d:
                return Path(root) / d

    return None


def find_summary_files(sample_dir):
    """Find all _mappedindelsummary.txt files in a sample directory.

    Args:
        sample_dir: Path to a sample directory.

    Returns:
        List of Path objects to summary files.
    """
    mapped_dir = sample_dir
    if not mapped_dir.exists():
        logger.warning(f"No mapped_reads directory in {sample_dir}")
        return []

    summary_files = []
    for subdir in sorted(mapped_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for f in subdir.iterdir():
            if f.is_dir():
                for file in f.iterdir():
                    if file.name.endswith('_processedindels.txt'):
                        summary_files.append(file)
            elif f.name.endswith('_processedindels.txt'):
                summary_files.append(f)

    logger.info(f"Found {len(summary_files)} summary files in {sample_dir.name}")
    return summary_files


def find_fasta_reference(sample_dir):
    """Find the expected target/PAM FASTA file for a sample directory.

    The FASTA file contains reference sequences for each oligo with
    headers in the format: >OligoID PAM_LOC PAM_DIR

    Args:
        sample_dir: Path to a sample directory.

    Returns:
        Path to the FASTA file, or None if not found.
    """
    # Look for exp_target_pam_*.fasta files
    data_parent = sample_dir.parent
    for pattern in ['exp_target_pam_new.fasta', 'exp_target_pam_old.fasta',
                    'exp_target_pam_both.fasta']:
        fasta = data_parent / pattern
        if fasta.exists():
            return fasta

    # Search more broadly
    for f in data_parent.rglob('exp_target_pam*.fasta'):
        return f

    return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dirs = find_trex2_dirs()
    for d in dirs:
        info = parse_sample_name(d.name)
        print(f"  {d.name}: cellline={info['cellline']}, dpi={info['dpi']}")
