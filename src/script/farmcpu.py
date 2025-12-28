# -*- coding: utf-8 -*-
"""
JanusX - FarmCPU GWAS Command-Line Interface

Design summary
--------------
  - High-memory implementation: loads full genotype into memory
  - Uses pyBLUP.QK for filtering and imputation
  - Two-step iterative method balancing power and false positives

Usage
-----
  jx farmcpu --vcf data.vcf.gz --pheno pheno.txt --out results
  jx farmcpu --bfile data --pheno pheno.txt --out results --qcov 5 --plot

Note: FarmCPU requires more memory than LM/LM due to full genotype loading.

Citation
--------
  https://github.com/MaizeMan-JxFU/JanusX/
"""

import os
import time
import socket
import argparse
import logging

# Matplotlib backend configuration (non-interactive, server-safe)
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
import psutil

from bioplotkit import GWASPLOT
from pyBLUP import QK
from gfreader import breader, vcfreader
from JanusX_rs.assoc import farmcpu
from ._common.log import setup_logging


def _section(logger: logging.Logger, title: str) -> None:
    """Pretty section separator in log."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def fastplot(
    gwasresult: pd.DataFrame,
    phenosub: np.ndarray,
    xlabel: str = "",
    outpdf: str = "fastplot.pdf",
) -> None:
    """Quick diagnostic plot: phenotype histogram, Manhattan, QQ."""
    results = gwasresult.astype({"POS": "int64"})
    fig = plt.figure(figsize=(16, 4), dpi=300)
    layout = [["A", "B", "B", "C"]]
    axes: dict[str, plt.Axes] = fig.subplot_mosaic(mosaic=layout)

    gwasplot = GWASPLOT(results)

    axes["A"].hist(phenosub, bins=15)
    axes["A"].set_xlabel(xlabel)
    axes["A"].set_ylabel("Count")

    gwasplot.manhattan(-np.log10(1 / results.shape[0]), ax=axes["B"])
    gwasplot.qq(ax=axes["C"])

    plt.tight_layout()
    plt.savefig(outpdf, transparent=True)


def load_phenotype(phenofile: str, ncol: list[int] | None, logger) -> pd.DataFrame:
    """Load and preprocess phenotype table."""
    logger.info(f"Loading phenotype from {phenofile}...")
    pheno = pd.read_csv(phenofile, sep="\t")
    pheno = pheno.groupby(pheno.columns[0]).mean()
    pheno.index = pheno.index.astype(str)

    if ncol is not None:
        assert np.min(ncol) < pheno.shape[1], "Phenotype column index out of range."
        ncol = [i for i in ncol if i in range(pheno.shape[1])]
        logger.info("Phenotypes: " + "\t".join(pheno.columns[ncol]))
        pheno = pheno.iloc[:, ncol]

    return pheno


def load_genotype_full(args, gfile: str, logger):
    """Load full genotype matrix into memory."""
    if args.vcf:
        logger.info(f"Loading genotype from {gfile}...")
        geno_df = vcfreader(gfile)
    elif args.bfile:
        logger.info(f"Loading genotype from {gfile}.bed...")
        geno_df = breader(gfile)
    else:
        raise ValueError("No genotype input for FarmCPU.")

    ref_alt = geno_df.iloc[:, :2]
    famid = geno_df.columns[2:].values.astype(str)
    geno = geno_df.iloc[:, 2:].to_numpy(copy=False)
    return ref_alt, famid, geno


def prepare_qk_and_filter(geno: np.ndarray, ref_alt: pd.DataFrame, logger):
    """Filter SNPs and impute using QK."""
    logger.info("* Filtering SNPs (MAF < 0.01 or missing rate > 0.05; mode imputation)...")
    logger.info("  Tip: Use pre-imputed genotypes from BEAGLE/IMPUTE2 if available.")
    qkmodel = QK(geno, maff=0.01)
    geno_filt = qkmodel.M

    ref_alt_filt = ref_alt.loc[qkmodel.SNPretain].copy()
    # Swap REF/ALT for very rare alleles
    ref_alt_filt.iloc[qkmodel.maftmark, [0, 1]] = ref_alt_filt.iloc[
        qkmodel.maftmark, [1, 0]
    ]
    ref_alt_filt["maf"] = qkmodel.maf
    logger.info("Filtering and imputation finished.")
    return geno_filt, ref_alt_filt, qkmodel


def build_qmatrix_farmcpu(
    gfile_prefix: str,
    qkmodel: QK,
    geno: np.ndarray,
    qdim: str,
    cov_path: str | None,
    logger,
) -> np.ndarray:
    """Build or load Q matrix for FarmCPU."""
    if qdim in np.arange(0, 30).astype(str):
        q_path = f"{gfile_prefix}.q.{qdim}.txt"
        if os.path.exists(q_path):
            logger.info(f"* Loading Q from {q_path}...")
            qmatrix = np.genfromtxt(q_path, dtype="float32")
        elif qdim == "0":
            qmatrix = np.array([]).reshape(geno.shape[1], 0)
        else:
            logger.info(f"* PCA dimension for Q: {qdim}")
            qmatrix, _eigval = qkmodel.PCA()
            qmatrix = qmatrix[:, : int(qdim)]
            np.savetxt(q_path, qmatrix, fmt="%.6f")
            logger.info(f"Cached Q written to {q_path}")
    else:
        logger.info(f"* Loading Q from {qdim}...")
        qmatrix = np.genfromtxt(qdim, dtype="float32")

    if cov_path:
        cov_arr = np.genfromtxt(cov_path, dtype=float)
        if cov_arr.ndim == 1:
            cov_arr = cov_arr.reshape(-1, 1)
        assert cov_arr.shape[0] == geno.shape[1], (
            f"Covariate rows ({cov_arr.shape[0]}) != samples ({geno.shape[1]})"
        )
        logger.info(f"Appending covariate: {cov_arr.shape}")
        qmatrix = np.concatenate([qmatrix, cov_arr], axis=1)

    logger.info(f"Q matrix shape: {qmatrix.shape}")
    return qmatrix


def run_farmcpu(
    args,
    gfile: str,
    prefix: str,
    logger,
    pheno: pd.DataFrame | None = None,
) -> None:
    """Run FarmCPU in high-memory mode."""
    phenofile = args.pheno
    outfolder = args.out
    qdim = args.qcov
    cov = args.cov

    t_loading = time.time()
    logger.info("* FarmCPU: loading genotype and phenotype")

    if pheno is None:
        pheno = load_phenotype(phenofile, args.ncol, logger)

    ref_alt, famid, geno = load_genotype_full(args, gfile, logger)
    logger.info(f"Loaded in {time.time() - t_loading:.2f}s")

    geno, ref_alt, qkmodel = prepare_qk_and_filter(geno, ref_alt, logger)
    assert geno.size > 0, "No SNPs after filtering."

    gfile_prefix = gfile.replace(".vcf", "").replace(".gz", "")

    qmatrix = build_qmatrix_farmcpu(
        gfile_prefix=gfile_prefix,
        qkmodel=qkmodel,
        geno=geno,
        qdim=qdim,
        cov_path=cov,
        logger=logger,
    )

    for phename in pheno.columns:
        logger.info(f"* FarmCPU for trait: {phename}")
        t_trait = time.time()

        p = pheno[phename].dropna()
        famidretain = np.isin(famid, p.index)
        if np.sum(famidretain) == 0:
            logger.info(f"Trait {phename}: no overlapping samples, skipped.")
            continue

        snp_sub = geno[:, famidretain]
        p_sub = p.loc[famid[famidretain]].values.reshape(-1, 1)
        q_sub = qmatrix[famidretain]

        logger.info(f"Samples: {np.sum(famidretain)}, SNPs: {snp_sub.shape[0]}")
        res = farmcpu(
            y=p_sub,
            M=snp_sub,
            X=q_sub,
            chrlist=ref_alt.reset_index().iloc[:, 0].values,
            poslist=ref_alt.reset_index().iloc[:, 1].values,
            iter=20,
            threads=args.thread,
        )
        res_df = pd.DataFrame(res, columns=["beta", "se", "p"], index=ref_alt.index)
        res_df = pd.concat([ref_alt, res_df], axis=1)
        res_df = res_df.reset_index()

        if args.plot:
            fastplot(
                res_df,
                p_sub,
                xlabel=phename,
                outpdf=f"{outfolder}/{prefix}.{phename}.farmcpu.pdf",
            )

        res_df = res_df.astype({"p": "object"})
        res_df.loc[:, "p"] = res_df["p"].map(lambda x: f"{x:.4e}")
        out_tsv = f"{outfolder}/{prefix}.{phename}.farmcpu.tsv"
        res_df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=None)
        logger.info(f"Saved: {out_tsv}")
        logger.info(f"Trait {phename} finished in {time.time() - t_trait:.2f}s")
        logger.info("")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    required_group = parser.add_argument_group("Required Arguments")
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument(
        "-vcf", "--vcf", type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz)",
    )
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help="Input genotype in PLINK binary format (prefix for .bed/.bim/.fam)",
    )
    required_group.add_argument(
        "-p", "--pheno", type=str, required=True,
        help="Phenotype file (tab-delimited, sample IDs in first column)",
    )

    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument(
        "-q", "--qcov", type=str, default="3",
        help="Number of PCs for Q matrix or path to Q file (default: 3)",
    )

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-n", "--ncol", action="extend", nargs="*",
        default=None, type=int,
        help="Zero-based phenotype column indices (e.g., '-n 0 -n 3')",
    )
    optional_group.add_argument(
        "-c", "--cov", type=str, default=None,
        help="Path to covariate file (aligned with genotype sample order)",
    )
    optional_group.add_argument(
        "-plot", "--plot", action="store_true", default=False,
        help="Generate diagnostic plots (default: %(default)s)",
    )
    optional_group.add_argument(
        "-t", "--thread", type=int, default=-1,
        help="CPU threads (-1=auto, default: %(default)s)",
    )
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory (default: %(default)s)",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Output file prefix (default: inferred)",
    )

    return parser.parse_args()


def main(log: bool = True):
    t_start = time.time()
    args = parse_args()

    if args.thread <= 0:
        args.thread = cpu_count()

    # Determine genotype source and prefix
    if args.vcf:
        gfile = args.vcf
        prefix = os.path.basename(gfile).replace(".gz", "").replace(".vcf", "")
    elif args.bfile:
        gfile = args.bfile
        prefix = os.path.basename(gfile)
    else:
        raise ValueError("No genotype input. Use -vcf or -bfile.")

    if args.prefix is not None:
        prefix = args.prefix

    gfile = gfile.replace("\\", "/")
    os.makedirs(args.out, 0o755, exist_ok=True)
    outprefix = f"{args.out}/{prefix}".replace("\\", "/").replace("//", "/")
    log_path = f"{outprefix}.farmcpu.log"
    logger = setup_logging(log_path)

    logger.info("JanusX - FarmCPU GWAS")
    logger.info(f"Host: {socket.gethostname()}\n")

    if log:
        logger.info("*" * 60)
        logger.info("FarmCPU GWAS CONFIGURATION")
        logger.info("*" * 60)
        logger.info(f"Genotype file:   {gfile}")
        logger.info(f"Phenotype file:  {args.pheno}")
        logger.info(f"Phenotype cols:  {args.ncol if args.ncol else 'All'}")
        logger.info(f"Q option:        {args.qcov}")
        if args.cov:
            logger.info(f"Covariate file:  {args.cov}")
        logger.info(f"Threads:         {args.thread}")
        logger.info(f"Output prefix:   {outprefix}")
        logger.info("*" * 60 + "\n")

    try:
        # Validate arguments
        q_is_valid = args.qcov in np.arange(0, 30).astype(str) or os.path.isfile(args.qcov)
        assert q_is_valid, f"{args.qcov} is invalid Q option."
        assert args.cov is None or os.path.isfile(args.cov), f"Covariate {args.cov} not found."

        _section(logger, "Run FarmCPU")
        run_farmcpu(
            args=args,
            gfile=gfile,
            prefix=prefix,
            logger=logger,
        )

    except Exception as e:
        logger.exception(f"Error in FarmCPU pipeline: {e}")

    lt = time.localtime()
    endinfo = (
        f"\nFinished. Total time: {round(time.time() - t_start, 2)}s\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()
