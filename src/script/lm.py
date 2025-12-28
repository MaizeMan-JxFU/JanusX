# -*- coding: utf-8 -*-
"""
JanusX - Linear Model (LM) GWAS Command-Line Interface

Design summary
--------------
  - Streaming implementation using rust2py.gfreader.load_genotype_chunks
  - Low-memory mode: processes genotype in chunks
  - No kinship matrix required

Usage
-----
  jx lm --vcf data.vcf.gz --pheno pheno.txt --out results
  jx lm --bfile data --pheno pheno.txt --out results --plot

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
from tqdm import tqdm
import psutil

from bioplotkit import GWASPLOT
from JanusX_rs.gfreader import load_genotype_chunks, inspect_genotype_file
from JanusX_rs.assoc import LM
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


def load_covariate(cov_path: str | None, n_samples: int, logger) -> np.ndarray | None:
    """Load covariate matrix aligned with genotype sample order."""
    if cov_path is None:
        return None

    logger.info(f"Loading covariate from {cov_path}...")
    cov_all = np.genfromtxt(cov_path, dtype="float32")
    if cov_all.ndim == 1:
        cov_all = cov_all.reshape(-1, 1)
    assert cov_all.shape[0] == n_samples, (
        f"Covariate rows ({cov_all.shape[0]}) != sample count ({n_samples})"
    )
    logger.info(f"Covariate shape: {cov_all.shape}")
    return cov_all


def run_lm_gwas(
    args,
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    n_snps: int,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    qmatrix: np.ndarray,
    cov_all: np.ndarray | None,
    logger: logging.Logger,
) -> None:
    """Run LM GWAS using streaming pipeline."""
    process = psutil.Process()
    n_cores = psutil.cpu_count(logical=True) or cpu_count()

    for pname in pheno.columns:
        logger.info(f"LM GWAS for trait: {pname}")

        cpu_t0 = process.cpu_times()
        rss0 = process.memory_info().rss
        t0 = time.time()
        peak_rss = rss0

        pheno_sub = pheno[pname].dropna()
        sameidx = np.isin(ids, pheno_sub.index)
        if np.sum(sameidx) == 0:
            logger.info(f"No overlapping samples for {pname}. Skipped.")
            continue

        y_vec = pheno_sub.loc[ids[sameidx]].values
        X_cov = qmatrix[sameidx]
        if cov_all is not None:
            X_cov = np.concatenate([X_cov, cov_all[sameidx]], axis=1)

        mod = LM(y=y_vec, X=X_cov)
        logger.info(f"Samples: {np.sum(sameidx)}, SNPs: {n_snps}")

        results_chunks = []
        info_chunks = []
        maf_list = []
        done_snps = 0

        process.cpu_percent(interval=None)
        pbar = tqdm(total=n_snps, desc=f"LM-{pname}", ascii=False)

        for genosub, sites in load_genotype_chunks(
            genofile, chunk_size, maf_threshold, max_missing_rate
        ):
            genosub = genosub[:, sameidx]
            maf_list.extend(np.mean(genosub, axis=1) / 2)

            results_chunks.append(mod.gwas(genosub, threads=args.thread))
            info_chunks.extend(
                [[s.chrom, s.pos, s.ref_allele, s.alt_allele] for s in sites]
            )

            m_chunk = genosub.shape[0]
            done_snps += m_chunk
            pbar.update(m_chunk)

            mem_info = process.memory_info()
            peak_rss = max(peak_rss, mem_info.rss)
            if done_snps % (10 * chunk_size) == 0:
                mem_gb = mem_info.rss / 1024**3
                pbar.set_postfix(memory=f"{mem_gb:.2f} GB")

        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

        cpu_t1 = process.cpu_times()
        rss1 = process.memory_info().rss
        t1 = time.time()

        wall = t1 - t0
        user_cpu = cpu_t1.user - cpu_t0.user
        sys_cpu = cpu_t1.system - cpu_t0.system
        total_cpu = user_cpu + sys_cpu
        avg_cpu_pct = 100.0 * total_cpu / wall / (n_cores or 1) if wall > 0 else 0.0
        avg_rss_gb = (rss0 + rss1) / 2 / 1024**3
        peak_rss_gb = peak_rss / 1024**3

        logger.info(
            f"SNP: {done_snps} | wall={wall:.2f}s, "
            f"CPU={avg_cpu_pct:.1f}%, avg RSS={avg_rss_gb:.2f}GB, peak≈{peak_rss_gb:.2f}GB"
        )

        if not results_chunks:
            logger.info(f"No SNPs for {pname}.")
            continue

        results = np.concatenate(results_chunks, axis=0)
        info_arr = np.array(info_chunks)

        df = pd.DataFrame(
            np.concatenate(
                [info_arr, results, np.array(maf_list).reshape(-1, 1)], axis=1
            ),
            columns=["#CHROM", "POS", "REF", "ALT", "beta", "se", "p", "maf"],
        )
        df = df[["#CHROM", "POS", "REF", "ALT", "maf", "beta", "se", "p"]]
        df = df.astype(
            {"POS": int, "maf": float, "beta": float, "se": float, "p": float}
        )

        if args.plot:
            fastplot(
                df, y_vec, xlabel=pname, outpdf=f"{outprefix}.{pname}.lm.pdf"
            )

        df = df.astype({"p": "object"})
        df.loc[:, "p"] = df["p"].map(lambda x: f"{x:.4e}")
        out_tsv = f"{outprefix}.{pname}.lm.tsv"
        df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=None)
        logger.info(f"Saved: {out_tsv}")
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

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-n", "--ncol", action="extend", nargs="*",
        default=None, type=int,
        help="Zero-based phenotype column indices (e.g., '-n 0 -n 3')",
    )
    optional_group.add_argument(
        "-q", "--qcov", type=str, default="0",
        help="Number of PCs for Q matrix or path to Q file (default: 0)",
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
        "-chunksize", "--chunksize", type=int, default=100_000,
        help="SNPs per chunk (default: %(default)s)",
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
    log_path = f"{outprefix}.lm.log"
    logger = setup_logging(log_path)

    logger.info("JanusX - Linear Model (LM) GWAS")
    logger.info(f"Host: {socket.gethostname()}\n")

    if log:
        logger.info("*" * 60)
        logger.info("LM GWAS CONFIGURATION")
        logger.info("*" * 60)
        logger.info(f"Genotype file:   {gfile}")
        logger.info(f"Phenotype file:  {args.pheno}")
        logger.info(f"Phenotype cols:  {args.ncol if args.ncol else 'All'}")
        logger.info(f"Q option:        {args.qcov}")
        if args.cov:
            logger.info(f"Covariate file:  {args.cov}")
        logger.info(f"Chunk size:      {args.chunksize}")
        logger.info(f"Threads:         {args.thread}")
        logger.info(f"Output prefix:   {outprefix}")
        logger.info("*" * 60 + "\n")

    try:
        # Load phenotype
        pheno = load_phenotype(args.pheno, args.ncol, logger)

        # Load genotype metadata
        ids, n_snps = inspect_genotype_file(gfile)
        ids = np.array(ids).astype(str)
        n_samples = len(ids)
        logger.info(f"Genotype: {n_samples} samples, {n_snps} SNPs")

        # Build/load Q matrix
        qmatrix = np.zeros((n_samples, 0), dtype="float32")
        if args.qcov != "0" and os.path.isfile(args.qcov):
            qmatrix = np.genfromtxt(args.qcov, dtype="float32")
        elif args.qcov not in ["0"]:
            dim = int(args.qcov)
            if dim > 0:
                logger.info(f"Q matrix: computing top {dim} PCs from genotype...")
                # Build GRM for PCA
                from .lmm import build_grm_streaming
                grm, _ = build_grm_streaming(
                    gfile, n_samples, n_snps, 0.01, 0.05, args.chunksize, 1, logger
                )
                _, eigvec = np.linalg.eigh(grm)
                qmatrix = eigvec[:, -dim:]
        logger.info(f"Q matrix shape: {qmatrix.shape}")

        # Load covariates
        cov_all = load_covariate(args.cov, n_samples, logger)

        # Run LM GWAS
        _section(logger, "Run LM GWAS")
        run_lm_gwas(
            args=args,
            genofile=gfile,
            pheno=pheno,
            ids=ids,
            n_snps=n_snps,
            outprefix=outprefix,
            maf_threshold=0.01,
            max_missing_rate=0.05,
            chunk_size=args.chunksize,
            qmatrix=qmatrix,
            cov_all=cov_all,
            logger=logger,
        )

    except Exception as e:
        logger.exception(f"Error in LM pipeline: {e}")

    lt = time.localtime()
    endinfo = (
        f"\nFinished. Total time: {round(time.time() - t_start, 2)}s\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()
