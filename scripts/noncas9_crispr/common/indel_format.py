"""
FORECasT indel string builder and parser.

Reimplements the indel string format from SelfTarget/selftarget/indel.py
to avoid import dependency issues while maintaining exact compatibility.

Format:
  Deletion:  D{size}_L{left}C{central}R{right}
  Insertion: I{size}_L{left}D{del_size}C{central}R{right}
  Wild-type: -

Where:
  - left:    position of last preserved base relative to cut site
  - right:   position of first preserved base on other side relative to cut site
  - central: microhomology ambiguity (bases that could belong to either side)
  - size:    number of deleted or inserted bases
"""

import re


def build_indel_string(del_size, left, right, ins_size=0, central=0):
    """Build a FORECasT-format indel identifier string.

    Args:
        del_size: Number of deleted bases (= right - left - 1 for pure deletions).
        left: Last preserved base position relative to cut site (typically negative).
        right: First preserved base position on the other side (typically positive).
        ins_size: Number of inserted bases (0 for pure deletions).
        central: Microhomology ambiguity count.

    Returns:
        Indel string, e.g. "D10_L-3C0R7" or "I2_L-1D0C0R0".
    """
    if ins_size > 0:
        return f"I{ins_size}_L{left}D{del_size}C{central}R{right}"
    return f"D{del_size}_L{left}C{central}R{right}"


def parse_indel_string(indel):
    """Parse a FORECasT-format indel string.

    Reimplements tokFullIndel from selftarget/indel.py (lines 4-23)
    for exact compatibility.

    Args:
        indel: Indel string, e.g. "D10_L-3C0R7" or "-" for wild-type.

    Returns:
        Tuple of (indel_type, indel_size, details_dict, mutations_list).
        - indel_type: 'D', 'I', or '-'
        - indel_size: size of the primary indel
        - details_dict: {'I': int, 'D': int, 'C': int, 'L': int, 'R': int}
        - mutations_list: list of (letter, position, nucleotides) tuples
    """
    indel_toks = indel.split('_')
    indel_type = indel_toks[0]
    indel_details = indel_toks[1] if len(indel_toks) > 1 else ''

    cigar_toks = re.findall(r'([CLRDI]+)(-?\d+)', indel_details)
    details = {'I': 0, 'D': 0, 'C': 0, 'L': 0, 'R': 0}
    for letter, val in cigar_toks:
        details[letter] = int(val)

    muts = []
    if len(indel_toks) > 2 or (indel_type == '-' and len(indel_toks) > 1):
        mut_toks = re.findall(r'([MNDSI]+)(-?\d+)(\[[ATGC]+\])?', indel_toks[-1])
        for letter, val, nucl in mut_toks:
            if nucl == '':
                nucl = '[]'
            muts.append((letter, int(val), nucl[1:-1]))

    if indel_type[0] == '-':
        isize = 0
    else:
        isize = int(indel_type[1:])

    return indel_type[0], isize, details, muts


def validate_indel_string(indel):
    """Check if an indel string is valid and parseable.

    Args:
        indel: Indel string to validate.

    Returns:
        True if parseable, False otherwise.
    """
    if indel == '-':
        return True
    try:
        itype, isize, details, muts = parse_indel_string(indel)
        if itype not in ('D', 'I', '-'):
            return False
        if itype == 'D' and isize <= 0:
            return False
        if itype == 'I' and isize <= 0:
            return False
        # For deletions: size + central = right - left - 1
        # (central accounts for microhomology ambiguity in deletion boundaries)
        if itype == 'D':
            span = details['R'] - details['L'] - 1
            if span != isize + details['C']:
                return False
        return True
    except Exception:
        return False


def indel_spans_cut_site(indel, cut_offset=0):
    """Check if an indel spans the cut site.

    Mirrors the filter in readSummaryToProfile: left <= 5, right >= -5.

    Args:
        indel: Indel string.
        cut_offset: Offset for cut site check (default 0, meaning left<=5, right>=-5).

    Returns:
        True if the indel spans the cut site.
    """
    if indel == '-':
        return True
    itype, isize, details, muts = parse_indel_string(indel)
    if itype == '-':
        return True
    return details['L'] <= 5 and details['R'] >= -5
