# -*- coding: utf-8 -*-
"""
JanusX - Linear Mixed Model (LMM) GWAS Command-Line Interface

Design summary
--------------
  - Streaming implementation using rust2py.gfreader.load_genotype_chunks
  - Low-memory mode: processes genotype in chunks
  - Requires kinship (GRM) matrix

Usage
-----
  jx lmm --vcf data.vcf.gz --pheno pheno.txt --out results
  jx lmm --bfile data --pheno pheno.txt --out results --grm 1 --qcov 3 --plot

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
from JanusX_rs.assoc import LMM
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


def build_grm_streaming(
    genofile: str,
    n_samples: int,
    n_snps: int,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    method: int,
    logger,
) -> tuple[np.ndarray, int]:
    """Build GRM in streaming fashion using rust2py.gfreader."""
    logger.info(f"Building GRM (streaming), method={method}")
    grm = np.zeros((n_samples, n_samples), dtype="float32")
    pbar = tqdm(total=n_snps, desc="GRM (streaming)", ascii=False)
    process = psutil.Process()

    varsum = 0.0
    eff_m = 0

    for genosub, _sites in load_genotype_chunks(
        genofile, chunk_size, maf_threshold, max_missing_rate
    ):
        genosub: np.ndarray
        maf = genosub.mean(axis=1, dtype="float32", keepdims=True) / 2
        genosub = genosub - 2 * maf

        if method == 1:
            grm += genosub.T @ genosub
            varsum += np.sum(2 * maf * (1 - maf))
        elif method == 2:
            w = 1.0 / (2 * maf * (1 - maf))
            grm += (genosub.T * w.ravel()) @ genosub
        else:
            raise ValueError(f"Unsupported GRM method: {method}")

        eff_m += genosub.shape[0]
        pbar.update(genosub.shape[0])

        if eff_m % (10 * chunk_size) == 0:
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f} GB")

    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()

    if method == 1:
        grm = (grm + grm.T) / varsum / 2
    else:
        grm = (grm + grm.T) / eff_m / 2

    logger.info("GRM construction finished.")
    return grm, eff_m


def load_or_build_grm_with_cache(
    genofile: str,
    prefix: str,
    mgrm: str,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, int]:
    """Load or build GRM with caching."""
    ids, n_snps = inspect_genotype_file(genofile)
    n_samples = len(ids)
    method_is_builtin = mgrm in ["1", "2"]

    if method_is_builtin:
        km_path = f"{prefix}.k.{mgrm}"
        if os.path.exists(f"{km_path}.npy"):
            logger.info(f"Loading cached GRM from {km_path}.npy...")
            grm = np.load(f"{km_path}.npy", mmap_mode="r")
            grm = grm.reshape(n_samples, n_samples)
            eff_m = n_snps
        else:
            method_int = int(mgrm)
            grm, eff_m = build_grm_streaming(
                genofile=genofile,
                n_samples=n_samples,
                n_snps=n_snps,
                maf_threshold=maf_threshold,
                max_missing_rate=max_missing_rate,
                chunk_size=chunk_size,
                method=method_int,
                logger=logger,
            )
            np.save(f"{km_path}.npy", grm)
            grm = np.load(f"{km_path}.npy", mmap_mode="r")
            logger.info(f"Cached GRM written to {km_path}.npy")
    else:
        assert os.path.isfile(mgrm), f"GRM file not found: {mgrm}"
        logger.info(f"Loading GRM from {mgrm}...")
        grm = np.genfromtxt(mgrm, dtype="float32")
        assert grm.size == n_samples * n_samples, "GRM size mismatch."
        grm = grm.reshape(n_samples, n_samples)
        eff_m = n_snps

    logger.info(f"GRM shape: {grm.shape}")
    return grm, eff_m


def build_pcs_from_grm(grm: np.ndarray, dim: int, logger: logging.Logger) -> np.ndarray:
    """Compute leading PCs from GRM."""
    logger.info(f"Computing top {dim} PCs from GRM...")
    _, eigvec = np.linalg.eigh(grm)
    pcs = eigvec[:, -dim:]
    logger.info("PC computation finished.")
    return pcs


def load_or_build_q_with_cache(
    grm: np.ndarray,
    prefix: str,
    pcdim: str,
    logger,
) -> np.ndarray:
    """Load or build Q matrix (PCs) with caching."""
    n = grm.shape[0]

    if pcdim in np.arange(1, n).astype(str):
        dim = int(pcdim)
        q_path = f"{prefix}.q.{pcdim}.txt"
        if os.path.exists(q_path):
            logger.info(f"Loading cached Q from {q_path}...")
            qmatrix = np.genfromtxt(q_path, dtype="float32")
        else:
            qmatrix = build_pcs_from_grm(grm, dim, logger)
            np.savetxt(q_path, qmatrix, fmt="%.6f")
            logger.info(f"Cached Q written to {q_path}")
    elif pcdim == "0":
        logger.info("PC dimension is 0; using empty Q matrix.")
        qmatrix = np.zeros((n, 0), dtype="float32")
    elif os.path.isfile(pcdim):
        logger.info(f"Loading Q from {pcdim}...")
        qmatrix = np.genfromtxt(pcdim, dtype="float32")
        assert qmatrix.shape[0] == n, "Q matrix row count mismatch."
    else:
        raise ValueError(f"Unknown Q option: {pcdim}")

    logger.info(f"Q matrix shape: {qmatrix.shape}")
    return qmatrix


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


def run_lmm_gwas(
    args,
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    n_snps: int,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    grm: np.ndarray,
    qmatrix: np.ndarray,
    cov_all: np.ndarray | None,
    eff_m: int,
    logger: logging.Logger,
) -> None:
    """Run LMM GWAS using streaming pipeline."""
    process = psutil.Process()
    n_cores = psutil.cpu_count(logical=True) or cpu_count()

    for pname in pheno.columns:
        logger.info(f"LMM GWAS for trait: {pname}")

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

        Ksub = grm[np.ix_(sameidx, sameidx)]
        mod = LMM(y=y_vec, X=X_cov, kinship=Ksub)
        logger.info(
            f"Samples: {np.sum(sameidx)}, SNPs: {eff_m}, PVE(null): {round(mod.pve, 3)}"
        )

        results_chunks = []
        info_chunks = []
        maf_list = []
        done_snps = 0

        process.cpu_percent(interval=None)
        pbar = tqdm(total=n_snps, desc=f"LMM-{pname}", ascii=False)

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
                df, y_vec, xlabel=pname, outpdf=f"{outprefix}.{pname}.lmm.pdf"
            )

        df = df.astype({"p": "object"})
        df.loc[:, "p"] = df["p"].map(lambda x: f"{x:.4e}")
        out_tsv = f"{outprefix}.{pname}.lmm.tsv"
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

    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument(
        "-k", "--grm", type=str, default="1",
        help="GRM: 1=centered, 2=standardized, or path to GRM file (default: 1)",
    )
    model_group.add_argument(
        "-q", "--qcov", type=str, default="0",
        help="Number of PCs for Q matrix or path to Q file (default: 0)",
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
    log_path = f"{outprefix}.lmm.log"
    logger = setup_logging(log_path)

    logger.info("JanusX - Linear Mixed Model (LMM) GWAS")
    logger.info(f"Host: {socket.gethostname()}\n")

    if log:
        logger.info("*" * 60)
        logger.info("LMM GWAS CONFIGURATION")
        logger.info("*" * 60)
        logger.info(f"Genotype file:   {gfile}")
        logger.info(f"Phenotype file:  {args.pheno}")
        logger.info(f"Phenotype cols:  {args.ncol if args.ncol else 'All'}")
        logger.info(f"GRM option:      {args.grm}")
        logger.info(f"Q option:        {args.qcov}")
        if args.cov:
            logger.info(f"Covariate file:  {args.cov}")
        logger.info(f"Chunk size:      {args.chunksize}")
        logger.info(f"Threads:         {args.thread}")
        logger.info(f"Output prefix:   {outprefix}")
        logger.info("*" * 60 + "\n")

    try:
        # Validate arguments
        grm_is_valid = args.grm in ["1", "2"] or os.path.isfile(args.grm)
        q_is_valid = args.qcov in np.arange(0, 30).astype(str) or os.path.isfile(args.qcov)
        assert grm_is_valid, f"{args.grm} is invalid GRM."
        assert q_is_valid, f"{args.qcov} is invalid Q option."
        assert args.cov is None or os.path.isfile(args.cov), f"Covariate {args.cov} not found."

        # Load phenotype
        pheno = load_phenotype(args.pheno, args.ncol, logger)

        # Load genotype metadata
        ids, n_snps = inspect_genotype_file(gfile)
        ids = np.array(ids).astype(str)
        n_samples = len(ids)
        logger.info(f"Genotype: {n_samples} samples, {n_snps} SNPs")

        # Build/load GRM with caching
        _section(logger, "Prepare GRM")
        grm, eff_m = load_or_build_grm_with_cache(
            genofile=gfile,
            prefix=outprefix,
            mgrm=args.grm,
            maf_threshold=0.01,
            max_missing_rate=0.05,
            chunk_size=args.chunksize,
            logger=logger,
        )

        # Build/load Q matrix with caching
        _section(logger, "Prepare Q matrix")
        qmatrix = load_or_build_q_with_cache(
            grm=grm,
            prefix=outprefix,
            pcdim=args.qcov,
            logger=logger,
        )

        # Load covariates
        cov_all = load_covariate(args.cov, n_samples, logger)

        # Run LMM GWAS
        _section(logger, "Run LMM GWAS")
        run_lmm_gwas(
            args=args,
            genofile=gfile,
            pheno=pheno,
            ids=ids,
            n_snps=n_snps,
            outprefix=outprefix,
            maf_threshold=0.01,
            max_missing_rate=0.05,
            chunk_size=args.chunksize,
            grm=grm,
            qmatrix=qmatrix,
            cov_all=cov_all,
            eff_m=eff_m,
            logger=logger,
        )

    except Exception as e:
        logger.exception(f"Error in LMM pipeline: {e}")

    lt = time.localtime()
    endinfo = (
        f"\nFinished. Total time: {round(time.time() - t_start, 2)}s\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()
