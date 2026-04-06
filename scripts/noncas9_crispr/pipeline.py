"""
Main pipeline orchestrator for non-Cas9 CRISPR data preprocessing.

CLI entry point: python -m src.pipeline --dataset [trex2|cas12a|all]

Steps: download -> parse -> convert -> validate -> summary stats
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from scripts.noncas9_crispr.config import (
    CAS12A_OUTPUT_DIR,
    CAS12A_RAW_DIR,
    FIGSHARE_RAW_DIR,
    INDEL_PROFILES_CSV,
    PROJECT_ROOT,
    REFERENCE_FASTA,
    TREX2_OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


def run_trex2_pipeline(skip_download=False):
    """Run the full TREX2 preprocessing pipeline.

    Args:
        skip_download: If True, skip the download step (assume data exists).

    Returns:
        dict with pipeline results.
    """
    logger.info("=" * 60)
    logger.info("TREX2 Pipeline")
    logger.info("=" * 60)

    results = {'dataset': 'trex2', 'timestamp': datetime.now().isoformat()}

    # Step 1: Download
    if not skip_download:
        logger.info("Step 1: Downloading FORECasT data from Figshare...")
        try:
            from scripts.noncas9_crispr.download.fetch_forecast import fetch_forecast_data
            fetch_forecast_data()
            results['download'] = 'success'
        except Exception as e:
            logger.error(f"Download failed: {e}")
            results['download'] = f'failed: {e}'
            return results
    else:
        logger.info("Step 1: Skipping download (--skip-download)")
        results['download'] = 'skipped'

    # Step 2: Extract and convert
    logger.info("Step 2: Converting TREX2 data to inDecay format...")
    try:
        from scripts.noncas9_crispr.trex2.convert import convert_trex2
        conv_results = convert_trex2()
        results['conversion'] = conv_results
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        results['conversion'] = {'success': False, 'error': str(e)}
        return results

    # Step 3: Validate
    logger.info("Step 3: Validating output...")
    try:
        from scripts.noncas9_crispr.common.validation import validate_output_pair
        csv_path = TREX2_OUTPUT_DIR / INDEL_PROFILES_CSV
        fasta_path = TREX2_OUTPUT_DIR / REFERENCE_FASTA
        val_results = validate_output_pair(csv_path, fasta_path)
        results['validation'] = {
            'valid': val_results['valid'],
            'csv_errors': val_results['csv']['errors'][:5],
            'fasta_errors': val_results['fasta']['errors'][:5],
            'cross_errors': val_results['cross_check_errors'],
        }
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        results['validation'] = {'valid': False, 'error': str(e)}

    logger.info(f"TREX2 pipeline complete. Valid: {results.get('validation', {}).get('valid', 'unknown')}")
    return results


def run_cas12a_pipeline(skip_download=False, orthologue='AsCas12a'):
    """Run the full Cas12a preprocessing pipeline.

    Args:
        skip_download: If True, skip the download step.
        orthologue: Cas12a orthologue to filter for.

    Returns:
        dict with pipeline results.
    """
    logger.info("=" * 60)
    logger.info("Cas12a Pipeline")
    logger.info("=" * 60)

    results = {'dataset': 'cas12a', 'timestamp': datetime.now().isoformat()}

    # Step 1: Download
    if not skip_download:
        logger.info("Step 1: Downloading Cas12a data...")
        try:
            from scripts.noncas9_crispr.download.fetch_cas12a import fetch_cas12a_supplementary
            fetch_cas12a_supplementary()
            results['download'] = 'success'
        except Exception as e:
            logger.error(f"Download failed: {e}")
            results['download'] = f'failed: {e}'

        # Check data availability
        from scripts.noncas9_crispr.download.fetch_cas12a import check_cas12a_data
        data_check = check_cas12a_data()
        if not data_check['available']:
            logger.error(
                "Cas12a data not available. Please download manually. "
                f"Place files in: {CAS12A_RAW_DIR}"
            )
            results['download'] = 'manual download required'
            return results
    else:
        logger.info("Step 1: Skipping download (--skip-download)")
        results['download'] = 'skipped'

    # Step 2: Convert
    logger.info("Step 2: Converting Cas12a data to inDecay format...")
    try:
        from scripts.noncas9_crispr.cas12a.convert import convert_cas12a
        conv_results = convert_cas12a(orthologue=orthologue)
        results['conversion'] = conv_results
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        results['conversion'] = {'success': False, 'error': str(e)}
        return results

    # Step 3: Validate
    logger.info("Step 3: Validating output...")
    try:
        from scripts.noncas9_crispr.common.validation import validate_output_pair
        csv_path = CAS12A_OUTPUT_DIR / INDEL_PROFILES_CSV
        fasta_path = CAS12A_OUTPUT_DIR / REFERENCE_FASTA
        if csv_path.exists():
            val_results = validate_output_pair(csv_path, fasta_path)
            results['validation'] = {
                'valid': val_results['valid'],
                'csv_errors': val_results['csv']['errors'][:5],
                'fasta_errors': val_results['fasta']['errors'][:5],
                'cross_errors': val_results['cross_check_errors'],
            }
        else:
            results['validation'] = {'valid': False, 'error': 'CSV file not generated'}
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        results['validation'] = {'valid': False, 'error': str(e)}

    logger.info(f"Cas12a pipeline complete. Valid: {results.get('validation', {}).get('valid', 'unknown')}")
    return results


def print_summary(results):
    """Print a human-readable summary of pipeline results."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    for result in results:
        dataset = result.get('dataset', 'unknown')
        print(f"\n--- {dataset.upper()} ---")
        print(f"  Timestamp: {result.get('timestamp', 'N/A')}")
        print(f"  Download:  {result.get('download', 'N/A')}")

        conv = result.get('conversion', {})
        if isinstance(conv, dict):
            print(f"  Conversion: {'SUCCESS' if conv.get('success') else 'FAILED'}")
            if conv.get('n_oligos_passing'):
                print(f"    Guides passing threshold: {conv['n_oligos_passing']}")
            if conv.get('n_reference_seqs'):
                print(f"    Reference sequences: {conv['n_reference_seqs']}")
            if conv.get('median_deletion_size') or conv.get('median_del_size'):
                med = conv.get('median_deletion_size') or conv.get('median_del_size')
                print(f"    Median deletion size: {med:.1f}bp")
            if conv.get('deletion_fraction'):
                print(f"    Deletion fraction: {conv['deletion_fraction']:.1%}")
            if conv.get('error'):
                print(f"    Error: {conv['error']}")

        val = result.get('validation', {})
        if isinstance(val, dict):
            print(f"  Validation: {'PASS' if val.get('valid') else 'FAIL'}")
            for err in val.get('csv_errors', [])[:3]:
                print(f"    CSV: {err}")
            for err in val.get('fasta_errors', [])[:3]:
                print(f"    FASTA: {err}")
            for err in val.get('cross_errors', [])[:3]:
                print(f"    Cross: {err}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess non-Cas9 CRISPR data for inDecay training',
    )
    parser.add_argument(
        '--dataset',
        choices=['trex2', 'cas12a', 'all'],
        default='all',
        help='Which dataset to process (default: all)',
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download (assume data already present)',
    )
    parser.add_argument(
        '--orthologue',
        default='AsCas12a',
        help='Cas12a orthologue to filter for (default: AsCas12a)',
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Write results to JSON file',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    results = []

    if args.dataset in ('trex2', 'all'):
        result = run_trex2_pipeline(skip_download=args.skip_download)
        results.append(result)

    if args.dataset in ('cas12a', 'all'):
        result = run_cas12a_pipeline(
            skip_download=args.skip_download,
            orthologue=args.orthologue,
        )
        results.append(result)

    # Print summary
    print_summary(results)

    # Write JSON output
    if args.output_json:
        output_path = Path(args.output_json)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results written to {output_path}")

    # Return exit code based on success
    all_valid = all(
        r.get('conversion', {}).get('success', False)
        for r in results
    )
    return 0 if all_valid else 1


if __name__ == '__main__':
    sys.exit(main())
