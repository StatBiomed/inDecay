"""
Build 60bp reference FASTA sequences for Cas12a guides.

Extracts target sequences and creates 60bp windows centered at the
chosen cut reference point (position 30 in the window).

FASTA header format: >GuideID 33 FORWARD
  - 33 = PAM_LOC so that cut_idx = 33 - 3 = 30
"""

import logging
from pathlib import Path

from scripts.noncas9_crispr.config import (
    CAS12A_CUT_REF_FROM_PAM,
    CAS12A_OLIGO_LEN,
    CAS12A_TARGET_LEN,
    INDECAY_CUT_POSITION,
    INDECAY_PAM_DIR,
    INDECAY_PAM_LOC,
    INDECAY_SEQ_LEN,
)

logger = logging.getLogger(__name__)


def extract_target_sequence(oligo_seq, target_start=None, target_len=None):
    """Extract the target region from a full oligo sequence.

    For Zhu et al. 2021, oligos are 142nt with a 42nt target region.

    Args:
        oligo_seq: Full oligo sequence.
        target_start: Start position of target in oligo. If None, auto-detect.
        target_len: Length of target region. Defaults to CAS12A_TARGET_LEN.

    Returns:
        Target sequence string.
    """
    if target_len is None:
        target_len = CAS12A_TARGET_LEN

    if target_start is not None:
        return oligo_seq[target_start:target_start + target_len]

    # If full oligo provided, extract central target region
    if len(oligo_seq) >= CAS12A_OLIGO_LEN:
        # Target is typically in the middle of the oligo
        start = (len(oligo_seq) - target_len) // 2
        return oligo_seq[start:start + target_len]

    # If sequence is already target-length or shorter, return as-is
    return oligo_seq


def build_60bp_reference(target_seq, cut_ref_from_pam=None):
    """Build a 60bp reference sequence centered at the cut site.

    The cut reference point (default: position 20 from PAM start) is
    placed at position 30 in the 60bp window.

    For Cas12a, PAM (TTTV) is at the 5' end of the target:
      PAM(4nt) + spacer(~20nt) + ... = target sequence
      Cut reference at position cut_ref_from_pam from PAM start

    Args:
        target_seq: Target/spacer region sequence (with PAM at 5' end).
        cut_ref_from_pam: Cut reference position from PAM start.

    Returns:
        60bp sequence string (padded with N if needed).
    """
    if cut_ref_from_pam is None:
        cut_ref_from_pam = CAS12A_CUT_REF_FROM_PAM

    # Cut reference in the target sequence coordinates
    cut_in_target = cut_ref_from_pam  # PAM starts at position 0

    # We want cut at position INDECAY_CUT_POSITION (30) in the 60bp window
    # So start = cut_in_target - 30
    start = cut_in_target - INDECAY_CUT_POSITION
    end = start + INDECAY_SEQ_LEN

    # Extract with padding
    if start < 0:
        left_pad = 'N' * (-start)
        seq = left_pad + target_seq[:end]
    else:
        seq = target_seq[start:end]

    # Pad right if needed
    if len(seq) < INDECAY_SEQ_LEN:
        seq += 'N' * (INDECAY_SEQ_LEN - len(seq))

    return seq[:INDECAY_SEQ_LEN]


def write_cas12a_fasta(guide_sequences, output_path, profiles=None):
    """Write Cas12a reference sequences to inDecay FASTA format.

    Args:
        guide_sequences: Dict mapping guide_id -> target_sequence.
        output_path: Output FASTA file path.
        profiles: Optional dict of profiles to filter guides by read count.

    Returns:
        Number of sequences written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0

    with open(output_path, 'w') as f:
        for guide_id in sorted(guide_sequences.keys()):
            target_seq = guide_sequences[guide_id]
            seq_60 = build_60bp_reference(target_seq)

            # Header: >GuideID 33 FORWARD
            # cut_idx = 33 - 3 = 30 (INDECAY_CUT_POSITION)
            f.write(f'>{guide_id} {INDECAY_PAM_LOC} {INDECAY_PAM_DIR}\n')
            f.write(f'{seq_60}\n')
            n_written += 1

    logger.info(f"Wrote {n_written} reference sequences to {output_path}")
    return n_written
