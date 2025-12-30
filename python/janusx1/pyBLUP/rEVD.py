from __future__ import annotations

from typing import Tuple, Literal
import os
import numpy as np

from ..janusx import grm_pca_bed, grm_pca_vcf


def _infer_input_kind(path_or_prefix: str) -> Literal["bed", "vcf"]:
    """
    Infer genotype input kind from a path/prefix.

    Rules (aligned with your load_genotype_chunks):
      1) If endswith .vcf or .vcf.gz -> "vcf"
      2) Else treat as PLINK prefix -> "bed", but verify prefix.bed/.bim/.fam exist
    """
    s = str(path_or_prefix)

    # VCF
    if s.endswith(".vcf") or s.endswith(".vcf.gz"):
        if not os.path.exists(s):
            raise FileNotFoundError(f"VCF file not found: {s}")
        return "vcf"

    # PLINK prefix
    bed = s + ".bed"
    bim = s + ".bim"
    fam = s + ".fam"
    missing = [p for p in (bed, bim, fam) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Input is treated as PLINK prefix but required files are missing:\n"
            + "\n".join(missing)
            + f"\n(prefix='{s}')"
        )
    return "bed"


def lrGRM(
    input_path: str,
    *,
    k: int = 20,
    oversample: int = 10,
    n_iter: int = 2,
    maf: float = 0.02,
    miss: float = 0.05,
    seed: int = 1,
    threads: int = 4,
    input_kind: Literal["auto", "bed", "vcf"] = "auto",
    return_float32: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomized PCA on the GRM (G = Z^T Z / den) using a Rust streaming kernel.

    This wraps Rust functions:
      - `grm_pca_bed(prefix, ...)` for PLINK bed/bim/fam
      - `grm_pca_vcf(path, ...)` for VCF/VCF.gz

    Parameters
    ----------
    input_path : str
        PLINK prefix (no extension) or VCF/VCF.GZ path.

    input_kind : {"auto","bed","vcf"}
        If "auto", infer by:
          - endswith .vcf/.vcf.gz -> vcf
          - otherwise             -> bed (and verify prefix.{bed,bim,fam} exist)

    (other params same as before)
    """
    # ---- Validate params ----
    if k <= 0:
        raise ValueError("k must be > 0")
    if oversample < 0:
        raise ValueError("oversample must be >= 0")
    if n_iter < 0:
        raise ValueError("n_iter must be >= 0")
    if threads <= 0:
        raise ValueError("threads must be >= 1")
    if not (0.0 <= maf <= 0.5):
        raise ValueError("maf must be in [0, 0.5]")
    if not (0.0 <= miss < 1.0):
        raise ValueError("miss must be in [0, 1)")

    # ---- Infer input kind ----
    if input_kind == "auto":
        input_kind = _infer_input_kind(input_path)
    elif input_kind in ("bed", "vcf"):
        # also do existence checks to fail fast
        _ = _infer_input_kind(input_path) if input_kind == "bed" else (
            None if (input_path.endswith(".vcf") or input_path.endswith(".vcf.gz")) else None
        )
        if input_kind == "vcf" and not os.path.exists(input_path):
            raise FileNotFoundError(f"VCF file not found: {input_path}")
    else:
        raise ValueError("input_kind must be 'auto', 'bed', or 'vcf'")

    # ---- Call Rust kernel ----
    if input_kind == "bed":
        evals, evecs = grm_pca_bed(
            input_path,
            int(k),
            int(oversample),
            int(n_iter),
            float(maf),
            float(miss),
            int(seed),
            int(threads),
        )
    else:  # vcf
        evals, evecs = grm_pca_vcf(
            input_path,
            int(k),
            int(oversample),
            int(n_iter),
            float(maf),
            float(miss),
            int(seed),
            int(threads),
        )

    # ---- Normalize outputs ----
    evals = np.ascontiguousarray(np.asarray(evals), dtype=np.float64).ravel()
    evecs = np.ascontiguousarray(np.asarray(evecs), dtype=np.float64)

    if return_float32:
        evals = evals.astype(np.float32, copy=False)
        evecs = evecs.astype(np.float32, copy=False)

    return evals, evecs
