# -*- coding: utf-8 -*-
'''
Examples:
  # Basic usage with VCF file
  --vcf genotypes.vcf --pheno phenotypes.txt --out results
  
  # Using PLINK binary files with custom parameters
  --bfile genotypes --pheno phenotypes.txt --out results --grm 1 --qcov 3 --thread 8
  
  # Using external kinship matrix and enabling fast mode
  --vcf genotypes.vcf --pheno phenotypes.txt --out results --grm kinship_matrix.txt --qcov 10 --fast
  
  # Maximum performance with all threads
  --bfile genotypes --pheno phenotypes.txt --out results --grm 1 --qcov 3 --cov covfile.txt --thread -1

File Formats:
  VCF/BFILE:    Standard VCF or PLINK binary format (bed/bim/fam)
  PHENO:        Tab-delimited file with sample IDs in first column and phenotypes in subsequent columns
  GRM File:     Space/tab-delimited kinship matrix file
  QCOV File:    Space/tab-delimited covariate matrix file
  COV File:    Space/tab-delimited covariate matrix file
        
Citation:
  https://github.com/MaizeMan-JxFU/gtools/
'''

from pyBLUP import GWAS
from pyBLUP import QK
from gfreader import breader,vcfreader,npyreader
import pandas as pd
import numpy as np
import argparse
import time
import socket
import os
from _common.log import setup_logging

def format_dataframe_for_export(df:pd.DataFrame, scientific_cols=None, float_cols=None):
    """
    Parameters:
    - df: raw DataFrame
    - scientific_cols: 科学计数法列
    - float_cols: 浮点数列
    """
    df_export = df.copy()
    # 科学计数法
    if scientific_cols:
        for col in scientific_cols:
            if col in df_export.columns and df_export[col].dtype in [np.float64, np.int64]:
                df_export[col] = df_export[col].apply(lambda x: f"{x:.4e}")
    # 浮点数
    if float_cols:
        for col in float_cols:
            if col in df_export.columns and df_export[col].dtype in [np.float64, np.int64]:
                df_export[col] = df_export[col].apply(lambda x: f"{x:.4f}")
    return df_export
def main(log:bool=True):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument('-vcf','--vcf', type=str, 
                           help='Input genotype file in VCF format (.vcf or .vcf.gz)')
    geno_group.add_argument('-bfile','--bfile', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .bed, .bim, .fam)')
    geno_group.add_argument('-npy','--npy', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .npz, .snp, .idv)')
    required_group.add_argument('-p','--pheno', type=str, required=True,
                               help='Phenotype file (tab-delimited with sample IDs in first column)')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('-k','--grm', type=str,
                               default='1',
                               help='Kinship matrix calculation method or path to pre-calculated GRM file '
                                   '(default: %(default)s)')
    optional_group.add_argument('-q','--qcov', type=str, default='3',
                               help='Number of principal components for Q matrix or path to covariate matrix file '
                                   '(default: %(default)s)')
    optional_group.add_argument('-c','--cov', type=str, default=None,
                               help='Path to Covariance file '
                                   '(default: %(default)s)')
    optional_group.add_argument('-t','--thread', type=int, default=-1,
                               help='Number of CPU threads to use (-1 for all available cores, default: %(default)s)')
    optional_group.add_argument('-fast','--fast', action='store_true', default=False,
                               help='Enable fast mode for GWAS (default: %(default)s)')
    optional_group.add_argument('-o', '--out', type=str, default='.',
                               help='Output directory for results'
                                   '(default: %(default)s)')
    args = parser.parse_args()
    # Determine genotype file
    if args.vcf:
        gfile = args.vcf
    elif args.bfile:
        gfile = args.bfile
    elif args.npy:
        gfile = args.npy
    # create output folder and log file
    os.makedirs(args.out,0o755,exist_ok=True)
    filename = os.path.basename(gfile)
    logger = setup_logging(f'''{args.out}/{filename.replace('.vcf','').replace('.gz','')}.log'''.replace('//','/'))
    # Print configuration summary
    logger.info('High Performance Linear Mixed Model Solver for Genome-Wide Association Studies')
    logger.info(f'Host: {socket.gethostname()}\n')
    if log:
        logger.info("*"*60)
        logger.info("GWAS LMM SOLVER CONFIGURATION")
        logger.info("*"*60)
        logger.info(f"Genotype file:    {gfile}")
        logger.info(f"Phenotype file:   {args.pheno}")
        logger.info(f"Output directory: {args.out}")
        logger.info(f"GRM method:       {args.grm}")
        logger.info(f"Q matrix:         {args.qcov}")
        logger.info(f"Covariant matrix: {args.cov}")
        logger.info(f"Threads:          {args.thread} ({'All cores' if args.thread == -1 else 'User specified'})")
        logger.info(f"FAST mode:        {args.fast}")
        logger.info("*"*60 + "\n")
    return gfile,args,logger

t_start = time.time()
gfile,args,logger = main()
phenofile,outfolder = args.pheno,args.out
kinship_method = args.grm
qdim = args.qcov
cov = args.cov
FASTmode = args.fast
threads = args.thread
kcal = True if kinship_method in ['1','2'] else False
qcal = True if qdim in np.arange(0,30).astype(str) else False

# test exist of all input files
assert os.path.isfile(phenofile), f"can not find file: {phenofile}"

if not kcal:
    assert os.path.isfile(kinship_method), f"{kinship_method} is not a GRM calculation method of kinship or a file"
if not qcal:
    assert os.path.isfile(qdim), f"{qdim} is not a dimension of q matrix or a file"
if cov is not None:
    assert os.path.isfile(cov), f"{cov} is not a file"

if not os.path.exists(outfolder):
    os.makedirs(outfolder,mode=0o755)
prefix = gfile.replace('.vcf','').replace('.gz','')

# Loading genotype matrix
logger.info(f'Loading phenotype from {phenofile}...')
pheno = pd.read_csv(rf'{phenofile}',sep='\t') # 第一列是样本ID, 第一行是表型名
pheno = pheno.groupby(pheno.columns[0]).mean() # 重复样本表型取均值
pheno.index = pheno.index.astype(str)
if pheno.shape[1]==0:
    print(pheno.head())
    raise ValueError('No phenotype data found, please check the phenotype file format!')
if args.vcf:
    logger.info(f'Loading genotype from {gfile}...')
    geno = vcfreader(rf'{gfile}') # VCF format
elif args.bfile:
    logger.info(f'Loading genotype from {gfile}.bed...')
    geno = breader(rf'{gfile}') # PLINK format
elif args.npy:
    logger.info(f'Loading genotype from {gfile}.npz...')
    geno = npyreader(rf'{gfile}') # numpy format
ref_alt:pd.DataFrame = geno.iloc[:,:2]
famid = geno.columns[2:].values.astype(str)
geno = geno.iloc[:,2:].to_numpy(copy=False)
logger.info('Geno and Pheno are ready!')

# GRM & PCA
qkmodel = QK(geno)
geno = qkmodel.M
ref_alt = ref_alt.loc[qkmodel.SNPretain]
ref_alt.iloc[qkmodel.maftmark,[0,1]] = ref_alt.iloc[qkmodel.maftmark,[1,0]]
ref_alt['maf'] = qkmodel.maf
if qcal or kcal:
    if not os.path.exists(f'{prefix}.k.{kinship_method}.txt') or not os.path.exists(f'{prefix}.q.{qdim}.txt') and int(qdim)!=0:
        logger.info(f'Samples and SNP: {geno.shape}')
    if os.path.exists(f'{prefix}.k.{kinship_method}.txt'):
        logger.info(f'* Loading GRM from {prefix}.k.{kinship_method}.txt...')
        kmatrix = np.genfromtxt(f'{prefix}.k.{kinship_method}.txt')
    else:    
        logger.info(f'* Calculation method of kinship matrix is {kinship_method}')
        kmatrix = qkmodel.GRM(method=int(kinship_method))
        np.savetxt(f'{prefix}.k.{kinship_method}.txt',kmatrix,fmt='%.6f')

    if os.path.exists(f'{prefix}.q.{qdim}.txt'):
        logger.info(f'* Loading Q matrix from {prefix}.q.{qdim}.txt...')
        qmatrix = np.genfromtxt(f'{prefix}.q.{qdim}.txt')
    else:
        if int(qdim) > 0:
            logger.info(f'* Dimension of PC for q matrix is {qdim}')
            qmatrix,eigenval = qkmodel.PCA()
            qmatrix = qmatrix[:,:int(qdim)]
            np.savetxt(f'{prefix}.q.{qdim}.txt',qmatrix,fmt='%.6f')
        else:
            qmatrix = np.array([]).reshape(geno.shape[0],0)
else:
    if qdim == '0':
        qmatrix = np.array([]).reshape(geno.shape[0],0)
    elif not qcal and os.path.exists(qdim):
        logger.info(f'* Loading Q matrix from {qdim}...')
        qmatrix = np.genfromtxt(qdim)
    
    if not kcal and os.path.exists(kinship_method):
        logger.info(f'* Loading GRM from {kinship_method}...')
        kmatrix = np.genfromtxt(kinship_method) if kinship_method[-4:] != '.npz' else np.load(kinship_method,)['arr_0']
if cov is not None:
    cov = np.genfromtxt(cov,).reshape(-1,1)
    logger.info(f'Covmatrix {cov.shape}:')
    qmatrix = np.concatenate([qmatrix,cov],axis=1)
logger.info(f'GRM {str(kmatrix.shape)}:')
logger.info(kmatrix[:5,:5])
logger.info(f'Qmatrix {str(qmatrix.shape)}:')
logger.info(qmatrix[:5,:5])
del qkmodel

# GWAS
for i in pheno.columns:
    t = time.time()
    logger.info('*'*60)
    p = pheno[i].dropna()
    famidretain = np.isin(famid,p.index)
    if len(p)>0:
        gwasmodel = GWAS(y=p.loc[famid[famidretain]].values.reshape(-1,1),X=qmatrix[famidretain],kinship=kmatrix[famidretain][:,famidretain])
        logger.info(f'''Phenotype: {i}, Number of samples: {np.sum(famidretain)}, Number of SNP: {geno.shape[0]}, pve of null: {round(gwasmodel.pve,3)}, FAST mode: {FASTmode}''')
        results = gwasmodel.gwas(snp=geno[:,famidretain],chunksize=100_000,threads=threads,fast=FASTmode) # gwas running...
        logger.info(f'Effective number of SNP: {results.shape[0]}')
        results = pd.DataFrame(results,columns=['beta','se','p'],index=ref_alt.index)
        results = pd.concat([ref_alt,results],axis=1)
        results = results.reset_index()
        results_save = format_dataframe_for_export(results, scientific_cols=['p'], float_cols=['beta','se','maf'])
        results_save.to_csv(f'{outfolder}/{i}.assoc.tsv',sep='\t',index=False)
        logger.info(f'Saved in {outfolder}/{i}.assoc.tsv'.replace('//','/'))
        del results,results_save,gwasmodel,p
    else:
        logger.info(f'Phenotype {i} has no overlapping samples with genotype, please check sample id. skipped.\n')
    logger.info(f'Time costed: {round(time.time()-t,2)} secs\n')
lt = time.localtime()
endinfo = f'\nFinished, Total time: {round(time.time()-t_start,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
logger.info(endinfo)