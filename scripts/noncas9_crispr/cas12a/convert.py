"""
Convert Cas12a data to inDecay CSV + FASTA format.

Orchestrates:
  1. Parse outcomes from Zhu et al. data
  2. Map to FORECasT indel strings
  3. Build reference FASTA
  4. Write inDecay-format outputs

Filters to a primary orthologue (AsCas12a) with option for all 16.
"""

import io
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.noncas9_crispr.cas12a.build_fasta import build_60bp_reference, write_cas12a_fasta
from scripts.noncas9_crispr.cas12a.indel_mapper import map_all_outcomes
from scripts.noncas9_crispr.cas12a.parse_outcomes import parse_cas12a_data
from scripts.noncas9_crispr.common.indel_format import parse_indel_string
from scripts.noncas9_crispr.config import (
    CAS12A_OUTPUT_DIR,
    CAS12A_RAW_DIR,
    INDEL_PROFILES_CSV,
    MIN_READS_THRESHOLD,
    REFERENCE_FASTA,
)

logger = logging.getLogger(__name__)

# AsCas12a is the primary/most-studied orthologue
PRIMARY_ORTHOLOGUE = 'AsCas12a'


def filter_by_orthologue(guide_outcomes, orthologue=None):
    """Filter guide outcomes by Cas12a orthologue.

    Args:
        guide_outcomes: Dict mapping guide_id -> outcomes.
        orthologue: Orthologue name to filter by (e.g. 'AsCas12a').
            If None, include all.

    Returns:
        Filtered dict.
    """
    if orthologue is None:
        return guide_outcomes

    filtered = {}
    for guide_id, outcomes in guide_outcomes.items():
        # Check if guide_id contains orthologue identifier
        if orthologue.lower() in guide_id.lower():
            filtered[guide_id] = outcomes

    if not filtered:
        logger.warning(
            f"No guides matched orthologue '{orthologue}'. "
            f"Available guide IDs (sample): {list(guide_outcomes.keys())[:5]}"
        )
        # Return all if filter matched nothing (user can debug)
        return guide_outcomes

    logger.info(
        f"Filtered to {len(filtered)} guides for orthologue '{orthologue}' "
        f"(from {len(guide_outcomes)} total)"
    )
    return filtered


def convert_outcomes_to_profiles(guide_outcomes, cut_ref_from_pam=None):
    """Convert parsed outcomes to FORECasT-format indel profiles.

    Args:
        guide_outcomes: Dict mapping guide_id -> list of outcome tuples.
        cut_ref_from_pam: Cut reference position for coordinate mapping.

    Returns:
        Dict mapping guide_id -> {indel_string: count}.
    """
    profiles = {}
    n_guides = 0
    n_skipped = 0

    for guide_id, outcomes in guide_outcomes.items():
        profile = map_all_outcomes(outcomes, cut_ref_from_pam=cut_ref_from_pam)

        # Apply read threshold
        total_reads = sum(profile.values())
        if total_reads < MIN_READS_THRESHOLD:
            n_skipped += 1
            continue

        profiles[guide_id] = profile
        n_guides += 1

    logger.info(
        f"Converted {n_guides} guides to profiles "
        f"({n_skipped} skipped below {MIN_READS_THRESHOLD}-read threshold)"
    )
    return profiles


def write_cas12a_csv(profiles, output_path):
    """Write Cas12a indel profiles to inDecay CSV format.

    Args:
        profiles: Dict mapping guide_id -> {indel_string: count}.
        output_path: Output CSV file path.

    Returns:
        Number of guides written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_oligos = 0
    n_rows = 0

    with io.open(output_path, 'w') as f:
        f.write('OligoID\tCount\tIdentifier\n')
        for guide_id in sorted(profiles.keys()):
            profile = profiles[guide_id]
            n_oligos += 1
            for indel, count in sorted(profile.items(), key=lambda x: -x[1]):
                if count <= 0:
                    continue
                f.write(f'{guide_id}\t{count}\t{indel}\n')
                n_rows += 1

    logger.info(f"Wrote {n_rows} rows for {n_oligos} guides to {output_path}")
    return n_oligos


def compute_cas12a_stats(profiles):
    """Compute summary statistics for Cas12a profiles.

    Args:
        profiles: Dict mapping guide_id -> {indel_string: count}.

    Returns:
        Dict of statistics.
    """
    all_del_sizes = []
    all_ins_sizes = []
    total_dels = 0
    total_ins = 0
    total_reads = 0

    for guide_id, profile in profiles.items():
        for indel, count in profile.items():
            total_reads += count
            try:
                itype, isize, details, muts = parse_indel_string(indel)
                if itype == 'D':
                    all_del_sizes.extend([isize] * count)
                    total_dels += count
                elif itype == 'I':
                    all_ins_sizes.extend([isize] * count)
                    total_ins += count
            except Exception:
                pass

    stats = {
        'n_guides': len(profiles),
        'total_reads': total_reads,
        'total_deletions': total_dels,
        'total_insertions': total_ins,
        'deletion_fraction': total_dels / max(total_reads, 1),
        'insertion_fraction': total_ins / max(total_reads, 1),
    }

    if all_del_sizes:
        stats['median_del_size'] = float(np.median(all_del_sizes))
        stats['mean_del_size'] = float(np.mean(all_del_sizes))
        stats['max_del_size'] = int(max(all_del_sizes))

    if all_ins_sizes:
        stats['median_ins_size'] = float(np.median(all_ins_sizes))

    return stats


def convert_cas12a(data_dir=None, output_dir=None, orthologue=None,
                    cut_ref_from_pam=None, guide_sequences=None):
    """Run the full Cas12a conversion pipeline.

    Args:
        data_dir: Directory with raw Cas12a data. Defaults to CAS12A_RAW_DIR.
        output_dir: Output directory. Defaults to CAS12A_OUTPUT_DIR.
        orthologue: Filter to specific orthologue (e.g. 'AsCas12a').
        cut_ref_from_pam: Cut reference position override.
        guide_sequences: Dict mapping guide_id -> target_sequence for FASTA.
            If None, FASTA generation is skipped with a warning.

    Returns:
        dict with conversion statistics.
    """
    data_dir = Path(data_dir) if data_dir else CAS12A_RAW_DIR
    output_dir = Path(output_dir) if output_dir else CAS12A_OUTPUT_DIR

    # Step 1: Parse outcomes
    logger.info(f"Parsing Cas12a data from {data_dir}...")
    guide_outcomes = parse_cas12a_data(data_dir)

    if not guide_outcomes:
        logger.error("No Cas12a outcomes parsed")
        return {
            'success': False,
            'error': 'No outcomes parsed. Check data files in ' + str(data_dir),
        }

    # Step 2: Filter by orthologue
    if orthologue:
        guide_outcomes = filter_by_orthologue(guide_outcomes, orthologue)

    # Step 3: Map to FORECasT format
    profiles = convert_outcomes_to_profiles(
        guide_outcomes, cut_ref_from_pam=cut_ref_from_pam,
    )

    if not profiles:
        logger.error("No guides passed read threshold after conversion")
        return {
            'success': False,
            'error': f'No guides passed {MIN_READS_THRESHOLD}-read threshold',
        }

    # Step 4: Write CSV
    csv_path = output_dir / INDEL_PROFILES_CSV
    n_oligos = write_cas12a_csv(profiles, csv_path)

    # Step 5: Build FASTA (if sequences available)
    fasta_path = output_dir / REFERENCE_FASTA
    n_seqs = 0
    if guide_sequences:
        # Filter to guides that passed threshold
        filtered_seqs = {
            gid: seq for gid, seq in guide_sequences.items()
            if gid in profiles
        }
        n_seqs = write_cas12a_fasta(filtered_seqs, fasta_path)
    else:
        logger.warning(
            "No guide sequences provided. FASTA file not generated. "
            "Provide guide_sequences dict to enable FASTA output."
        )

    # Step 6: Compute stats
    stats = compute_cas12a_stats(profiles)
    stats.update({
        'success': True,
        'n_oligos_passing': n_oligos,
        'n_reference_seqs': n_seqs,
        'csv_path': str(csv_path),
        'fasta_path': str(fasta_path),
        'orthologue': orthologue or 'all',
    })

    logger.info(f"Cas12a conversion complete: {stats}")
    return stats


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    convert_cas12a()
