# -*- coding: utf-8 -*-
"""
JanusX - High Performance GWAS Command-Line Interface

This module is a unified wrapper that dispatches to specialized GWAS implementations:
  - lm.py   : Linear Model (LM) - streaming, low-memory
  - lmm.py  : Linear Mixed Model (LMM) - streaming, low-memory, requires kinship
  - farmcpu.py : FarmCPU - high-memory, full genotype loading

Usage
-----
  # Run all models
  jx gwas --vcf data.vcf.gz --pheno pheno.txt --lm --lmm --farmcpu --out results

  # Or run individual models directly
  jx lm   --vcf data.vcf.gz --pheno pheno.txt --out results
  jx lmm  --vcf data.vcf.gz --pheno pheno.txt --out results
  jx farmcpu --vcf data.vcf.gz --pheno pheno.txt --out results

For detailed help on each model:
  jx lm -h
  jx lmm -h
  jx farmcpu -h

Citation
--------
  https://github.com/MaizeMan-JxFU/JanusX/
"""

import sys
import os
import time
import socket
import argparse
import logging

from ._common.log import setup_logging


def _section(logger: logging.Logger, title: str) -> None:
    """Pretty section separator in log."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def determine_genotype_source(args) -> tuple[str, str]:
    """Determine genotype input path and output prefix from CLI arguments."""
    if args.vcf:
        gfile = args.vcf
        prefix = os.path.basename(gfile).replace(".gz", "").replace(".vcf", "")
    elif args.bfile:
        gfile = args.bfile
        prefix = os.path.basename(gfile)
    else:
        raise ValueError("No genotype input specified. Use -vcf or -bfile.")
    if args.prefix is not None:
        prefix = args.prefix
    gfile = gfile.replace("\\", "/")
    return gfile, prefix


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    required_group = parser.add_argument_group("Required arguments")

    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument(
        "-vcf", "--vcf", type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz)",
    )
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help="Input genotype in PLINK binary format "
             "(prefix for .bed, .bim, .fam)",
    )

    required_group.add_argument(
        "-p", "--pheno", type=str, required=True,
        help="Phenotype file (tab-delimited, sample IDs in the first column)",
    )

    models_group = parser.add_argument_group("Model Arguments")
    models_group.add_argument(
        "-lmm", "--lmm", action="store_true", default=False,
        help="Run linear mixed model (low-memory, chunk-based) "
             "(default: %(default)s)",
    )
    models_group.add_argument(
        "-farmcpu", "--farmcpu", action="store_true", default=False,
        help="Run FarmCPU model (high-memory, full genotype) "
             "(default: %(default)s)",
    )
    models_group.add_argument(
        "-lm", "--lm", action="store_true", default=False,
        help="Run general linear model (low-memory, chunk-based) "
             "(default: %(default)s)",
    )

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-n", "--ncol", action="extend", nargs="*",
        default=None, type=int,
        help='Zero-based phenotype column indices to analyze. '
             'E.g., "-n 0 -n 3" to analyze the 1st and 4th traits '
             "(default: %(default)s)",
    )
    optional_group.add_argument(
        "-k", "--grm", type=str, default="1",
        help="GRM option: 1 (centering), 2 (standardization), "
             "or path to precomputed GRM file (default: %(default)s)",
    )
    optional_group.add_argument(
        "-q", "--qcov", type=str, default="0",
        help="Number of principal components for Q matrix or path to Q file "
             "(default: %(default)s)",
    )
    optional_group.add_argument(
        "-c", "--cov", type=str, default=None,
        help="Path to additional covariate file. "
             "For LMM/LM, the file must be aligned with the genotype sample "
             "order from inspect_genotype_file (one row per sample). "
             "For FarmCPU, it must follow the genotype sample order "
             "(famid) (default: %(default)s)",
    )
    optional_group.add_argument(
        "-plot", "--plot", action="store_true", default=False,
        help="Generate diagnostic plots (histogram, Manhattan, QQ) "
             "(default: %(default)s)",
    )
    optional_group.add_argument(
        "-chunksize", "--chunksize", type=int, default=100_000,
        help="Number of SNPs per chunk for streaming LMM/LM "
             "(affects GRM and GWAS; default: %(default)s)",
    )
    optional_group.add_argument(
        "-t", "--thread", type=int, default=-1,
        help="Number of CPU threads (-1 uses all available cores, "
             "default: %(default)s)",
    )
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for results (default: %(default)s)",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix for output files (default: %(default)s)",
    )

    return parser.parse_args()


def main(log: bool = True):
    """
    Unified GWAS wrapper that dispatches to lm, lmm, or farmcpu sub-modules.

    This function parses arguments and delegates to the appropriate specialized
    implementation based on which model flags are specified.
    """
    from joblib import cpu_count
    import numpy as np

    t_start = time.time()
    args = parse_args()

    gfile, prefix = determine_genotype_source(args)

    os.makedirs(args.out, 0o755, exist_ok=True)
    outprefix = f"{args.out}/{prefix}".replace("\\", "/").replace("//", "/")
    log_path = f"{outprefix}.gwas.log"
    logger = setup_logging(log_path)

    logger.info(
        "JanusX - GWAS CLI (unified wrapper for LM/LMM/FarmCPU)"
    )
    logger.info(f"Host: {socket.gethostname()}\n")

    if log:
        logger.info("*" * 60)
        logger.info("GWAS CONFIGURATION")
        logger.info("*" * 60)
        logger.info(f"Genotype file:    {gfile}")
        logger.info(f"Phenotype file:   {args.pheno}")
        logger.info(f"Phenotype cols:   {args.ncol if args.ncol is not None else 'All'}")
        logger.info(
            f"Models:           "
            f"{'LMM ' if args.lmm else ''}"
            f"{'LM ' if args.lm else ''}"
            f"{'FarmCPU' if args.farmcpu else ''}"
        )
        logger.info(f"GRM option:       {args.grm}")
        logger.info(f"Q option:         {args.qcov}")
        if args.cov:
            logger.info(f"Covariate file:   {args.cov}")
        logger.info(f"Chunk size:       {args.chunksize}")
        logger.info(f"Threads:          {args.thread} ({cpu_count()} available)")
        logger.info(f"Output prefix:    {outprefix}")
        logger.info("*" * 60 + "\n")

    try:
        # Validate that at least one model is selected
        assert (args.lm or args.lmm or args.farmcpu), (
            "No model selected. Use --lm, --lmm, and/or --farmcpu. "
            "For individual help: jx lm -h | jx lmm -h | jx farmcpu -h"
        )

        # Import sub-modules
        from . import lm, lmm as lmm_module, farmcpu as farmcpu_module

        # --- Dispatch to LM ---
        if args.lm:
            _section(logger, "Dispatching to LM (Linear Model)")
            # Modify args for LM module (LM uses --qcov, not --grm)
            import argparse as argparse_alias
            lm_args = argparse_alias.Namespace(
                vcf=args.vcf,
                bfile=args.bfile,
                pheno=args.pheno,
                ncol=args.ncol,
                qcov=args.qcov,
                cov=args.cov,
                plot=args.plot,
                chunksize=args.chunksize,
                thread=args.thread if args.thread > 0 else cpu_count(),
                out=args.out,
                prefix=prefix,
            )
            lm.main(log=log)

        # --- Dispatch to LMM ---
        if args.lmm:
            _section(logger, "Dispatching to LMM (Linear Mixed Model)")
            import argparse as argparse_alias
            lmm_args = argparse_alias.Namespace(
                vcf=args.vcf,
                bfile=args.bfile,
                pheno=args.pheno,
                ncol=args.ncol,
                grm=args.grm,
                qcov=args.qcov,
                cov=args.cov,
                plot=args.plot,
                chunksize=args.chunksize,
                thread=args.thread if args.thread > 0 else cpu_count(),
                out=args.out,
                prefix=prefix,
            )
            lmm_module.main(log=log)

        # --- Dispatch to FarmCPU ---
        if args.farmcpu:
            _section(logger, "Dispatching to FarmCPU (high-memory)")
            import argparse as argparse_alias
            farmcpu_args = argparse_alias.Namespace(
                vcf=args.vcf,
                bfile=args.bfile,
                pheno=args.pheno,
                ncol=args.ncol,
                qcov=args.qcov,
                cov=args.cov,
                plot=args.plot,
                thread=args.thread if args.thread > 0 else cpu_count(),
                out=args.out,
                prefix=prefix,
            )
            farmcpu_module.main(log=log)

    except ImportError as e:
        logger.exception(f"Failed to import sub-module: {e}")
        logger.info("Ensure lm.py, lmm.py, and farmcpu.py exist in the script directory.")
    except Exception as e:
        logger.exception(f"Error in JanusX GWAS pipeline: {e}")

    lt = time.localtime()
    endinfo = (
        f"\nFinished. Total wall time: {round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()