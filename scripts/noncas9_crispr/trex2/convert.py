"""
Convert TREX2 FORECasT data to inDecay CSV + FASTA format.

Reads _mappedindelsummary.txt files and produces:
  - indel_profiles.csv: OligoID, Count, Identifier (tab-delimited)
  - reference.fasta: 60bp sequences centered at cut site

Follows readSummaryToProfile logic from selftarget/profile.py:175-243.
"""

import csv
import io
import logging
from collections import defaultdict
from pathlib import Path

from scripts.noncas9_crispr.common.indel_format import parse_indel_string, validate_indel_string
from scripts.noncas9_crispr.config import (
    FIGSHARE_RAW_DIR,
    INDECAY_CUT_POSITION,
    INDECAY_PAM_DIR,
    INDECAY_PAM_LOC,
    INDECAY_SEQ_LEN,
    MIN_READS_THRESHOLD,
    TREX2_OUTPUT_DIR,
    TREX2_REFERENCE_FILE,
    INDEL_PROFILES_CSV,
    REFERENCE_FASTA,
)
from scripts.noncas9_crispr.trex2.extract_samples import (
    find_summary_files,
    find_trex2_dirs,
)

logger = logging.getLogger(__name__)


def read_summary_to_profile(filename, oligoid=None):
    """Read a mapped indel summary file into a profile dict.

    Simplified version of readSummaryToProfile from profile.py:175-243.
    Skips WT subtraction (handled separately) but applies cut-site filter.

    Args:
        filename: Path to _mappedindelsummary.txt file.
        oligoid: If set, only read this oligo's data.

    Returns:
        Dict mapping OligoID -> {indel_string: read_count}.
    """
    profiles = defaultdict(lambda: defaultdict(int))
    filename = Path(filename)

    if not filename.exists():
        logger.warning(f"Summary file not found: {filename}")
        return profiles

    with io.open(filename) as f:
        reader = csv.reader(f, delimiter='\t')
        curr_oligo_id = None

        for toks in reader:
            if not toks:
                continue

            # OligoID header line
            if toks[0][:3] == '@@@':
                curr_oligo_id = toks[0][3:].split()[0]
                continue

            if oligoid is not None and oligoid != curr_oligo_id:
                continue

            if curr_oligo_id is None:
                continue
            
            indel = toks[0]
            oligo_indel = toks[0]
            indel_seq = toks[2]
            try:
                num_reads = int(toks[1])
            except (ValueError, IndexError):
                continue

            # Filter: skip reads with problematic oligo indels
            # if oligo_indel != '-':
            #     # Only allow oligo indels that are small and outside guide/PAM
            #     # Simplified filter: skip any non-trivial oligo indels
            #     try:
            #         itype, isize, details, muts = parse_indel_string(oligo_indel)
            #         if itype != '-':
            #             if isize > 2 or (details['L'] < 6 and details['R'] > -20):
            #                 continue
            #         # Check mutations in guide/PAM region
            #         mut_locs = [x for x in muts if x[0] not in ['N', 'I', 'D']]
            #         if len(mut_locs) > 0:
            #             if any(x[1] > -20 and x[1] < 6 for x in mut_locs):
            #                 continue
            #             if len(mut_locs) > 5:
            #                 continue
            #         ins_del_muts = [x for x in muts if x[0] in ['I', 'D']]
            #         if ins_del_muts and any(x[1] > 2 for x in ins_del_muts):
            #             continue
            #     except Exception:
            #         continue

            # # Filter: only indels spanning the cut site
            # if indel != '-':
            #     try:
            #         itype, isize, details, muts = parse_indel_string(indel)
            #         if itype != '-' and (details['L'] > 5 or details['R'] < -5):
            #             continue
            #     except Exception:
            #         continue

            profiles[curr_oligo_id][indel] += num_reads

    return profiles


def aggregate_profiles(sample_dirs):
    """Aggregate indel profiles across replicate TREX2 directories.

    Sums read counts per indel per OligoID across all replicates.

    Args:
        sample_dirs: List of TREX2 sample directory paths.

    Returns:
        Dict mapping OligoID -> {indel_string: total_read_count}.
    """
    aggregated = defaultdict(lambda: defaultdict(int))

    for sample_dir in sample_dirs:
        summary_files = find_summary_files(sample_dir)
        logger.info(f"Processing {sample_dir.name}: {len(summary_files)} summary files")

        for summary_file in summary_files:
            profiles = read_summary_to_profile(summary_file)
            for oligo_id, profile in profiles.items():
                for indel, count in profile.items():
                    aggregated[oligo_id][indel] += count

    logger.info(f"Aggregated profiles for {len(aggregated)} oligos")
    return aggregated


def load_reference_sequences(reference_path=None):
    """Load reference sequences from the TREX2 reference TSV file.

    Reads the reference.txt file (Supplementary Table 19 from Allen et al. 2019)
    and extracts 60bp centered at cut site for each oligo.

    The TSV columns are: ID, Guide Sequence, TargetSequence, Scaffold, Subset,
    Comments, PAM Index, Strand.

    Args:
        reference_path: Path to reference.txt. Defaults to FIGSHARE_RAW_DIR / TREX2_REFERENCE_FILE.

    Returns:
        Dict mapping OligoID -> (sequence_60bp, pam_loc, pam_dir).
    """
    if reference_path is None:
        reference_path = FIGSHARE_RAW_DIR / TREX2_REFERENCE_FILE
    reference_path = Path(reference_path)

    if not reference_path.exists():
        logger.error(f"Reference file not found: {reference_path}")
        return {}

    references = {}

    with io.open(reference_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            oligo_id = row['ID']
            seq = row['TargetSequence']
            pam_loc = int(row['PAM Index'])
            pam_dir = row['Strand']

            # Calculate cut site in the original sequence
            if pam_dir == 'FORWARD':
                cut_site = pam_loc - 3
            else:
                cut_site = pam_loc + 3

            # Extract 60bp centered at cut site (cut at position 30)
            start = cut_site - INDECAY_CUT_POSITION
            end = start + INDECAY_SEQ_LEN

            if start < 0:
                seq_60 = 'N' * (-start) + seq[:end]
            elif end > len(seq):
                seq_60 = seq[start:] + 'N' * (end - len(seq))
            else:
                seq_60 = seq[start:end]

            # Ensure exactly 60bp
            seq_60 = seq_60[:INDECAY_SEQ_LEN]
            if len(seq_60) < INDECAY_SEQ_LEN:
                seq_60 += 'N' * (INDECAY_SEQ_LEN - len(seq_60))

            if oligo_id not in references:
                references[oligo_id] = (seq_60, pam_loc, pam_dir)

    logger.info(f"Loaded reference sequences for {len(references)} oligos")
    return references


def write_indecay_csv(profiles, output_path):
    """Write indel profiles to inDecay CSV format.

    Args:
        profiles: Dict mapping OligoID -> {indel_string: count}.
        output_path: Output CSV file path.

    Returns:
        Number of oligos written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_oligos = 0
    n_rows = 0

    with io.open(output_path, 'w') as f:
        f.write('OligoID\tCount\tIdentifier\n')
        for oligo_id in sorted(profiles.keys()):
            profile = profiles[oligo_id]

            # Apply read threshold
            total_reads = sum(profile.values())
            if total_reads < MIN_READS_THRESHOLD:
                continue

            n_oligos += 1
            for indel, count in sorted(profile.items(), key=lambda x: -x[1]):
                if indel == '-':
                    continue  # Skip wild-type reads
                if count <= 0:
                    continue
                f.write(f'{oligo_id}\t{count}\t{indel}\n')
                n_rows += 1

    logger.info(f"Wrote {n_rows} rows for {n_oligos} oligos to {output_path}")
    return n_oligos


def write_indecay_fasta(references, profiles, output_path):
    """Write reference sequences to inDecay FASTA format.

    Only includes oligos that pass the read threshold in profiles.

    Args:
        references: Dict mapping OligoID -> (seq_60bp, pam_loc, pam_dir).
        profiles: Dict mapping OligoID -> {indel: count} (for filtering).
        output_path: Output FASTA file path.

    Returns:
        Number of sequences written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0

    with io.open(output_path, 'w') as f:
        for oligo_id in sorted(references.keys()):
            # Only include oligos that pass the read threshold
            if oligo_id in profiles:
                total_reads = sum(profiles[oligo_id].values())
                if total_reads < MIN_READS_THRESHOLD:
                    continue
            else:
                continue

            seq_60, orig_pam_loc, orig_pam_dir = references[oligo_id]

            # Write with standardized header: cut at position 30
            # PAM_LOC = 33 -> cut_idx = 33 - 3 = 30
            f.write(f'>{oligo_id} {INDECAY_PAM_LOC} {INDECAY_PAM_DIR}\n')
            f.write(f'{seq_60}\n')
            n_written += 1

    logger.info(f"Wrote {n_written} reference sequences to {output_path}")
    return n_written


def convert_trex2(data_root=None, output_dir=None):
    """Run the full TREX2 conversion pipeline.

    Args:
        data_root: Root directory with Figshare data. Defaults to FIGSHARE_RAW_DIR.
        output_dir: Output directory. Defaults to TREX2_OUTPUT_DIR.

    Returns:
        dict with conversion statistics.
    """
    output_dir = Path(output_dir) if output_dir else TREX2_OUTPUT_DIR

    # Find TREX2 directories
    trex2_dirs = find_trex2_dirs(data_root)
    if not trex2_dirs:
        logger.error("No TREX2 DPI7 directories found")
        return {'success': False, 'error': 'No TREX2 directories found'}

    # Load reference sequences from the downloaded reference.txt
    references = load_reference_sequences()

    # Aggregate profiles across replicates
    profiles = aggregate_profiles(trex2_dirs)

    # Write outputs
    csv_path = output_dir / INDEL_PROFILES_CSV
    fasta_path = output_dir / REFERENCE_FASTA

    n_oligos = write_indecay_csv(profiles, csv_path)
    n_seqs = write_indecay_fasta(references, profiles, fasta_path)

    # Compute statistics
    all_del_sizes = []
    for oligo_id, profile in profiles.items():
        total = sum(profile.values())
        if total < MIN_READS_THRESHOLD:
            continue
        for indel, count in profile.items():
            if indel == '-':
                continue
            try:
                itype, isize, details, muts = parse_indel_string(indel)
                if itype == 'D':
                    all_del_sizes.extend([isize] * count)
            except Exception:
                pass

    import numpy as np
    stats = {
        'success': True,
        'n_trex2_dirs': len(trex2_dirs),
        'n_oligos_passing': n_oligos,
        'n_reference_seqs': n_seqs,
        'csv_path': str(csv_path),
        'fasta_path': str(fasta_path),
    }
    if all_del_sizes:
        stats['median_deletion_size'] = float(np.median(all_del_sizes))
        stats['mean_deletion_size'] = float(np.mean(all_del_sizes))
        stats['max_deletion_size'] = int(max(all_del_sizes))

    logger.info(f"TREX2 conversion complete: {stats}")
    return stats


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    convert_trex2()
