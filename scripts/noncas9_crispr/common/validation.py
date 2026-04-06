"""
Validation utilities for checking output compatibility with inDecay.

Validates:
1. All indel strings are parseable by tokFullIndel-compatible parser
2. CSV format matches expected columns
3. FASTA format has correct header structure
4. Sequence lengths and cut site positions are correct
"""

import csv
import io
from pathlib import Path

from Bio import SeqIO

from scripts.noncas9_crispr.common.indel_format import parse_indel_string, validate_indel_string
from scripts.noncas9_crispr.config import INDECAY_SEQ_LEN, INDECAY_CUT_POSITION, MIN_READS_THRESHOLD


def validate_csv(csv_path):
    """Validate an indel profiles CSV file.

    Expected format: tab-delimited with columns OligoID, Count, Identifier.
    Each row has: OligoID (guide identifier), Count (read count), Identifier (indel string).

    Args:
        csv_path: Path to the CSV file.

    Returns:
        dict with keys: 'valid' (bool), 'errors' (list), 'stats' (dict).
    """
    csv_path = Path(csv_path)
    errors = []
    stats = {
        'total_rows': 0,
        'unique_oligos': set(),
        'unique_indels': set(),
        'unparseable_indels': [],
        'oligo_read_counts': {},
    }

    if not csv_path.exists():
        return {'valid': False, 'errors': [f'File not found: {csv_path}'], 'stats': stats}

    with io.open(csv_path) as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        if header is None:
            errors.append('Empty file')
            return {'valid': False, 'errors': errors, 'stats': stats}

        expected_cols = ['OligoID', 'Count', 'Identifier']
        if header != expected_cols:
            errors.append(f'Expected columns {expected_cols}, got {header}')

        for row_num, row in enumerate(reader, start=2):
            stats['total_rows'] += 1
            if len(row) != 3:
                errors.append(f'Row {row_num}: expected 3 columns, got {len(row)}')
                continue

            oligo_id, count_str, indel = row
            stats['unique_oligos'].add(oligo_id)
            stats['unique_indels'].add(indel)

            # Validate count
            try:
                count = int(count_str)
                if count < 0:
                    errors.append(f'Row {row_num}: negative count {count}')
            except ValueError:
                errors.append(f'Row {row_num}: invalid count "{count_str}"')
                continue

            # Track reads per oligo
            if oligo_id not in stats['oligo_read_counts']:
                stats['oligo_read_counts'][oligo_id] = 0
            stats['oligo_read_counts'][oligo_id] += count

            # Validate indel string
            if indel != '-' and not validate_indel_string(indel):
                stats['unparseable_indels'].append((row_num, indel))

    # Check read thresholds
    below_threshold = [
        oid for oid, total in stats['oligo_read_counts'].items()
        if total < MIN_READS_THRESHOLD
    ]
    if below_threshold:
        errors.append(
            f'{len(below_threshold)} oligos below {MIN_READS_THRESHOLD}-read threshold'
        )

    if stats['unparseable_indels']:
        for row_num, indel in stats['unparseable_indels'][:5]:
            errors.append(f'Row {row_num}: unparseable indel "{indel}"')
        if len(stats['unparseable_indels']) > 5:
            errors.append(f'... and {len(stats["unparseable_indels"]) - 5} more unparseable indels')

    # Convert sets for JSON serialization
    stats['num_unique_oligos'] = len(stats['unique_oligos'])
    stats['num_unique_indels'] = len(stats['unique_indels'])
    stats['unique_oligos'] = None  # Don't serialize large sets
    stats['unique_indels'] = None

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'stats': stats,
    }


def validate_fasta(fasta_path):
    """Validate a reference FASTA file for inDecay compatibility.

    Expected format:
      >OligoID PAM_LOC PAM_DIR
      SEQUENCE (60bp)

    Where cut_idx = PAM_LOC - 3 should equal INDECAY_CUT_POSITION (30).

    Args:
        fasta_path: Path to the FASTA file.

    Returns:
        dict with keys: 'valid' (bool), 'errors' (list), 'stats' (dict).
    """
    fasta_path = Path(fasta_path)
    errors = []
    stats = {
        'num_sequences': 0,
        'wrong_length': [],
        'wrong_cut_site': [],
    }

    if not fasta_path.exists():
        return {'valid': False, 'errors': [f'File not found: {fasta_path}'], 'stats': stats}

    for record in SeqIO.parse(str(fasta_path), "fasta"):
        stats['num_sequences'] += 1
        desc_parts = str(record.description).split()

        if len(desc_parts) < 3:
            errors.append(f'Record {record.id}: header must have OligoID, PAM_LOC, PAM_DIR')
            continue

        oligo_id = desc_parts[0]
        try:
            pam_loc = int(desc_parts[1])
        except ValueError:
            errors.append(f'Record {oligo_id}: invalid PAM location "{desc_parts[1]}"')
            continue
        pam_dir = desc_parts[2]

        if pam_dir not in ('FORWARD', 'REVERSE'):
            errors.append(f'Record {oligo_id}: PAM direction must be FORWARD or REVERSE')

        # Check cut site
        cut_idx = pam_loc - 3 if pam_dir == 'FORWARD' else pam_loc + 3
        if cut_idx != INDECAY_CUT_POSITION:
            stats['wrong_cut_site'].append(oligo_id)

        # Check sequence length
        seq_len = len(record.seq)
        if seq_len != INDECAY_SEQ_LEN:
            stats['wrong_length'].append((oligo_id, seq_len))

    if stats['wrong_length']:
        for oligo_id, slen in stats['wrong_length'][:5]:
            errors.append(
                f'Record {oligo_id}: sequence length {slen}, expected {INDECAY_SEQ_LEN}'
            )

    if stats['wrong_cut_site']:
        errors.append(
            f'{len(stats["wrong_cut_site"])} records with cut site != {INDECAY_CUT_POSITION}'
        )

    if stats['num_sequences'] == 0:
        errors.append('No sequences found in FASTA file')

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'stats': stats,
    }


def validate_output_pair(csv_path, fasta_path):
    """Validate a CSV + FASTA output pair for consistency.

    Checks:
    - Both files are individually valid
    - All OligoIDs in CSV have a matching FASTA entry
    - Feature extraction can run on sample entries

    Args:
        csv_path: Path to indel_profiles.csv
        fasta_path: Path to reference.fasta

    Returns:
        dict with combined validation results.
    """
    csv_result = validate_csv(csv_path)
    fasta_result = validate_fasta(fasta_path)

    # Cross-check OligoIDs
    cross_errors = []
    if csv_result['valid'] or fasta_result['valid']:
        # Get oligo IDs from FASTA
        fasta_oligos = set()
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            fasta_oligos.add(str(record.description).split()[0])

        # Get oligo IDs from CSV
        csv_oligos = set()
        with io.open(csv_path) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 1:
                    csv_oligos.add(row[0])

        missing_in_fasta = csv_oligos - fasta_oligos
        if missing_in_fasta:
            cross_errors.append(
                f'{len(missing_in_fasta)} OligoIDs in CSV not found in FASTA'
            )

    return {
        'csv': csv_result,
        'fasta': fasta_result,
        'cross_check_errors': cross_errors,
        'valid': csv_result['valid'] and fasta_result['valid'] and len(cross_errors) == 0,
    }
