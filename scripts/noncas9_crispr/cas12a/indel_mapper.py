"""
Map Cas12a indel outcomes to FORECasT indel string format.

THE CORE CHALLENGE: Cas12a has a staggered cut (4-5nt 5' overhangs)
unlike Cas9's blunt cuts. We use a configurable cut reference point
(default: midpoint of the stagger at ~position 20 from PAM start).

Cas12a cut geometry:
  - Non-target strand (NTS): ~18nt from PAM
  - Target strand (TS): ~23nt from PAM
  - Creates 4-5nt 5' overhangs

Cut reference:
  - Midpoint: position 20 from PAM start (default)
  - NTS: position 18 from PAM start
  - TS: position 23 from PAM start
"""

import logging

from scripts.noncas9_crispr.common.indel_format import build_indel_string, validate_indel_string
from scripts.noncas9_crispr.config import CAS12A_CUT_REF_FROM_PAM

logger = logging.getLogger(__name__)


def cas12a_to_forecast_indel(del_start_from_pam, del_len, ins_seq="",
                              cut_ref_from_pam=None):
    """Convert a Cas12a deletion to FORECasT indel string format.

    Coordinate system:
      - Position 0 = 5' end of PAM (TTTV)
      - del_start_from_pam = first deleted base position
      - del_len = number of deleted bases
      - cut_ref_from_pam = reference cut position from PAM start

    The FORECasT format uses coordinates relative to the cut site:
      - left = position of last preserved base - cut_ref
      - right = position of first preserved base after deletion - cut_ref

    Args:
        del_start_from_pam: Start position of deletion from PAM (0-based).
        del_len: Length of deletion in bases.
        ins_seq: Inserted sequence (empty string for pure deletions).
        cut_ref_from_pam: Cut reference position from PAM. Defaults to config value.

    Returns:
        FORECasT indel string, e.g. "D10_L-3C0R7".
    """
    if cut_ref_from_pam is None:
        cut_ref_from_pam = CAS12A_CUT_REF_FROM_PAM

    # Last preserved base before the deletion (0-indexed from PAM)
    last_preserved = del_start_from_pam - 1

    # First preserved base after the deletion (0-indexed from PAM)
    first_preserved_after = del_start_from_pam + del_len

    # Convert to coordinates relative to cut reference
    left = last_preserved - cut_ref_from_pam
    right = first_preserved_after - cut_ref_from_pam

    del_size = del_len  # Should equal right - left - 1
    central = 0  # No microhomology ambiguity info available

    # Validate consistency
    assert del_size == right - left - 1, (
        f"Inconsistent deletion: del_size={del_size}, "
        f"right-left-1={right - left - 1}"
    )

    ins_size = len(ins_seq) if ins_seq else 0
    return build_indel_string(del_size, left, right, ins_size=ins_size, central=central)


def cas12a_insertion_to_forecast(ins_pos_from_pam, ins_seq,
                                  del_start_from_pam=None, del_len=0,
                                  cut_ref_from_pam=None):
    """Convert a Cas12a insertion (with optional deletion) to FORECasT format.

    Args:
        ins_pos_from_pam: Insertion position from PAM.
        ins_seq: Inserted nucleotide sequence.
        del_start_from_pam: If there's also a deletion, its start position.
        del_len: Length of accompanying deletion (0 if pure insertion).
        cut_ref_from_pam: Cut reference position from PAM.

    Returns:
        FORECasT indel string.
    """
    if cut_ref_from_pam is None:
        cut_ref_from_pam = CAS12A_CUT_REF_FROM_PAM

    if del_start_from_pam is not None and del_len > 0:
        # Insertion with deletion (complex indel)
        return cas12a_to_forecast_indel(
            del_start_from_pam, del_len,
            ins_seq=ins_seq,
            cut_ref_from_pam=cut_ref_from_pam,
        )
    else:
        # Pure insertion: left = ins_pos - 1 - cut_ref, right = ins_pos - cut_ref
        left = ins_pos_from_pam - 1 - cut_ref_from_pam
        right = ins_pos_from_pam - cut_ref_from_pam
        ins_size = len(ins_seq)
        return build_indel_string(0, left, right, ins_size=ins_size, central=0)


def map_all_outcomes(guide_outcomes, cut_ref_from_pam=None):
    """Map all Cas12a outcomes for a guide to FORECasT format.

    Args:
        guide_outcomes: List of outcome tuples from parse_outcomes.
            Each tuple is either:
              (del_start, del_len, read_count) for deletions
              (ins_pos, ins_seq, read_count) for insertions

        cut_ref_from_pam: Cut reference position.

    Returns:
        Dict mapping FORECasT indel string -> total read count.
    """
    profile = {}
    n_mapped = 0
    n_failed = 0

    for outcome in guide_outcomes:
        if len(outcome) < 3:
            n_failed += 1
            continue

        val1, val2, count = outcome
        if count is None:
            count = 1

        try:
            if isinstance(val2, str):
                # Insertion: (pos, seq, count)
                indel_str = cas12a_insertion_to_forecast(
                    val1, val2, cut_ref_from_pam=cut_ref_from_pam,
                )
            elif isinstance(val2, int):
                # Deletion: (start, length, count)
                indel_str = cas12a_to_forecast_indel(
                    val1, val2, cut_ref_from_pam=cut_ref_from_pam,
                )
            else:
                n_failed += 1
                continue

            # Validate the generated indel string
            if not validate_indel_string(indel_str):
                logger.debug(f"Generated invalid indel string: {indel_str}")
                n_failed += 1
                continue

            if indel_str not in profile:
                profile[indel_str] = 0
            profile[indel_str] += count
            n_mapped += 1

        except Exception as e:
            logger.debug(f"Failed to map outcome {outcome}: {e}")
            n_failed += 1

    if n_failed > 0:
        logger.debug(f"Mapped {n_mapped} outcomes, failed {n_failed}")

    return profile
