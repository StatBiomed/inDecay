"""
Parse Cas12a indel outcome data from Zhu et al. 2021.

Expected input format: tables with indel outcomes as
[DEL_start_pos, DEL_len]:read_counts

Position 0 = 5' end of PAM (TTTV).
"""

import csv
import io
import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def parse_del_entry(entry_str):
    """Parse a single deletion entry string.

    Args:
        entry_str: String like "[5, 10]" or "[5,10]:250"

    Returns:
        Tuple of (del_start, del_len, read_count) or None if unparseable.
    """
    entry_str = entry_str.strip()

    # Try format: [start, len]:count
    match = re.match(r'\[(\d+),\s*(\d+)\]\s*:\s*(\d+)', entry_str)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))

    # Try format: [start, len] (count provided separately)
    match = re.match(r'\[(\d+),\s*(\d+)\]', entry_str)
    if match:
        return int(match.group(1)), int(match.group(2)), None

    return None


def parse_ins_entry(entry_str):
    """Parse a single insertion entry string.

    Args:
        entry_str: String like "[5, 'ACG']" or "[5, 'ACG']:100"

    Returns:
        Tuple of (ins_pos, ins_seq, read_count) or None if unparseable.
    """
    entry_str = entry_str.strip()

    # Try format: [pos, 'seq']:count
    match = re.match(r"\[(\d+),\s*'([ATGC]+)'\]\s*:\s*(\d+)", entry_str)
    if match:
        return int(match.group(1)), match.group(2), int(match.group(3))

    # Try format: [pos, 'seq']
    match = re.match(r"\[(\d+),\s*'([ATGC]+)'\]", entry_str)
    if match:
        return int(match.group(1)), match.group(2), None

    return None


def parse_outcome_dict_string(dict_str):
    """Parse a Python-dict-like string of indel outcomes.

    Args:
        dict_str: String like "{[5, 10]: 250, [8, 3]: 100}"

    Returns:
        List of (del_start, del_len, read_count) tuples.
    """
    outcomes = []

    if not dict_str or dict_str.strip() in ('{}', 'nan', '', 'None'):
        return outcomes

    # Remove outer braces
    dict_str = dict_str.strip()
    if dict_str.startswith('{'):
        dict_str = dict_str[1:]
    if dict_str.endswith('}'):
        dict_str = dict_str[:-1]

    # Split on comma-separated entries (careful with commas inside brackets)
    # Pattern: [num, num]: num
    entries = re.findall(r'\[([^\]]+)\]\s*:\s*(\d+)', dict_str)
    for bracket_content, count in entries:
        parts = bracket_content.split(',')
        if len(parts) == 2:
            try:
                start = int(parts[0].strip())
                length = int(parts[1].strip())
                outcomes.append((start, length, int(count)))
            except ValueError:
                # Might be an insertion: [pos, 'seq']
                seq_match = re.match(r"\s*'([ATGC]+)'\s*", parts[1])
                if seq_match:
                    try:
                        pos = int(parts[0].strip())
                        outcomes.append((pos, seq_match.group(1), int(count)))
                    except ValueError:
                        pass

    return outcomes


def parse_cas12a_excel(filepath):
    """Parse Cas12a outcome data from an Excel supplementary file.

    Attempts to identify and parse tables with guide IDs and indel outcomes.

    Args:
        filepath: Path to Excel file.

    Returns:
        Dict mapping guide_id -> list of outcome tuples.
    """
    filepath = Path(filepath)
    guide_outcomes = defaultdict(list)

    try:
        # Read all sheets
        xlsx = pd.ExcelFile(filepath)
        for sheet_name in xlsx.sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet_name)

            # Look for columns that might contain guide IDs and outcomes
            guide_col = None
            outcome_col = None
            count_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                if any(x in col_lower for x in ['guide', 'target', 'grna', 'sgrna', 'crrna']):
                    guide_col = col
                if any(x in col_lower for x in ['deletion', 'del', 'indel', 'outcome', 'mutation']):
                    outcome_col = col
                if any(x in col_lower for x in ['count', 'reads', 'frequency', 'freq']):
                    count_col = col

            if guide_col is None or outcome_col is None:
                continue

            logger.info(
                f"Sheet '{sheet_name}': guide_col='{guide_col}', "
                f"outcome_col='{outcome_col}', count_col='{count_col}'"
            )

            for _, row in df.iterrows():
                guide_id = str(row[guide_col]).strip()
                if guide_id in ('nan', '', 'None'):
                    continue

                outcome_str = str(row[outcome_col])
                outcomes = parse_outcome_dict_string(outcome_str)

                if not outcomes and count_col is not None:
                    # Try parsing outcome as single entry with count in separate column
                    parsed = parse_del_entry(outcome_str)
                    if parsed:
                        del_start, del_len, _ = parsed
                        try:
                            count = int(row[count_col])
                        except (ValueError, TypeError):
                            count = 1
                        outcomes = [(del_start, del_len, count)]

                for outcome in outcomes:
                    guide_outcomes[guide_id].append(outcome)

    except Exception as e:
        logger.error(f"Error parsing Excel file {filepath}: {e}")

    logger.info(f"Parsed outcomes for {len(guide_outcomes)} guides from {filepath.name}")
    return dict(guide_outcomes)


def parse_cas12a_csv(filepath):
    """Parse Cas12a outcome data from a CSV/TSV file.

    Args:
        filepath: Path to CSV/TSV file.

    Returns:
        Dict mapping guide_id -> list of outcome tuples.
    """
    filepath = Path(filepath)
    guide_outcomes = defaultdict(list)

    sep = '\t' if filepath.suffix in ('.tsv', '.txt') else ','

    try:
        df = pd.read_csv(filepath, sep=sep)
        # Use same column detection logic as Excel parser
        guide_col = None
        outcome_col = None

        for col in df.columns:
            col_lower = str(col).lower()
            if any(x in col_lower for x in ['guide', 'target', 'grna', 'sgrna', 'crrna']):
                guide_col = col
            if any(x in col_lower for x in ['deletion', 'del', 'indel', 'outcome', 'mutation']):
                outcome_col = col

        if guide_col is None or outcome_col is None:
            logger.warning(f"Could not identify guide/outcome columns in {filepath.name}")
            logger.info(f"Available columns: {list(df.columns)}")
            return dict(guide_outcomes)

        for _, row in df.iterrows():
            guide_id = str(row[guide_col]).strip()
            if guide_id in ('nan', '', 'None'):
                continue

            outcome_str = str(row[outcome_col])
            outcomes = parse_outcome_dict_string(outcome_str)
            for outcome in outcomes:
                guide_outcomes[guide_id].append(outcome)

    except Exception as e:
        logger.error(f"Error parsing CSV file {filepath}: {e}")

    logger.info(f"Parsed outcomes for {len(guide_outcomes)} guides from {filepath.name}")
    return dict(guide_outcomes)


def parse_cas12a_data(data_dir):
    """Parse all Cas12a data files in a directory.

    Automatically detects file format (Excel or CSV/TSV).

    Args:
        data_dir: Directory containing Cas12a data files.

    Returns:
        Dict mapping guide_id -> list of outcome tuples.
    """
    data_dir = Path(data_dir)
    all_outcomes = defaultdict(list)

    for f in sorted(data_dir.rglob('*')):
        if f.suffix in ('.xlsx', '.xls'):
            outcomes = parse_cas12a_excel(f)
        elif f.suffix in ('.csv', '.tsv', '.txt'):
            outcomes = parse_cas12a_csv(f)
        else:
            continue

        for guide_id, guide_outcomes in outcomes.items():
            all_outcomes[guide_id].extend(guide_outcomes)

    logger.info(f"Total: parsed outcomes for {len(all_outcomes)} guides")
    return dict(all_outcomes)
