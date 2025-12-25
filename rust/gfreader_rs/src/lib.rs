use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

use memmap2::Mmap;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::Bound;

// ============================================================
// Variant metadata
// ============================================================

/// Variant metadata: chrom, pos, ref, alt.
#[pyclass]
#[derive(Clone)]
pub struct SiteInfo {
    #[pyo3(get)]
    pub chrom: String,
    #[pyo3(get)]
    pub pos: i32,
    #[pyo3(get)]
    pub ref_allele: String,
    #[pyo3(get)]
    pub alt_allele: String,
}

// ============================================================
// PLINK helpers: read .fam/.bim
// ============================================================

/// Read PLINK .fam and return sample IDs (IID, the 2nd column).
fn read_fam(prefix: &str) -> Result<Vec<String>, String> {
    let fam_path = format!("{prefix}.fam");
    let file = File::open(&fam_path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);

    let mut samples = Vec::new();
    for line in reader.lines() {
        let l = line.map_err(|e| e.to_string())?;
        let mut it = l.split_whitespace();
        it.next(); // FID
        if let Some(iid) = it.next() {
            samples.push(iid.to_string());
        } else {
            return Err(format!("Malformed FAM line: {l}"));
        }
    }
    Ok(samples)
}

/// Read PLINK .bim and return SiteInfo list.
fn read_bim(prefix: &str) -> Result<Vec<SiteInfo>, String> {
    let bim_path = format!("{prefix}.bim");
    let file = File::open(&bim_path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);

    let mut sites = Vec::new();
    for line in reader.lines() {
        let l = line.map_err(|e| e.to_string())?;
        let cols: Vec<&str> = l.split_whitespace().collect();
        if cols.len() < 6 {
            return Err(format!("Malformed BIM line: {l}"));
        }
        let chrom = cols[0].to_string();
        let pos: i32 = cols[3].parse().unwrap_or(0);
        let a1 = cols[4].to_string();
        let a2 = cols[5].to_string();

        sites.push(SiteInfo {
            chrom,
            pos,
            ref_allele: a1,
            alt_allele: a2,
        });
    }
    Ok(sites)
}

// ============================================================
// SNP row processing: missing filter + MAF filter + flip + mean impute
// ============================================================

/// Process a single SNP row (float32 genotypes), applying:
///  - missing rate filter
///  - MAF filter
///  - flip alleles if ALT freq > 0.5 (swap ref/alt and g = 2 - g)
///  - mean imputation for missing values (keeps float, may be non-integer)
///
/// Missing is represented as g < 0 (e.g., -9.0).
///
/// Returns:
///   true  -> keep this SNP
///   false -> filter out
fn process_snp_row(
    row: &mut [f32],
    ref_allele: &mut String,
    alt_allele: &mut String,
    maf_threshold: f32,
    max_missing_rate: f32,
) -> bool {
    let mut alt_sum: f64 = 0.0;
    let mut non_missing: i64 = 0;

    for &g in row.iter() {
        if g >= 0.0 {
            alt_sum += g as f64;
            non_missing += 1;
        }
    }

    let n_samples = row.len() as f64;
    if n_samples == 0.0 {
        return false;
    }

    // Missing rate filter
    let missing_rate = 1.0 - (non_missing as f64 / n_samples);
    if missing_rate > max_missing_rate as f64 {
        return false;
    }

    if non_missing == 0 {
        // All missing but not exceeding max_missing_rate:
        // ALT freq and MAF are effectively 0.
        // If maf_threshold > 0, filter; otherwise fill 0 and keep.
        if maf_threshold > 0.0 {
            return false;
        } else {
            for g in row.iter_mut() {
                *g = 0.0;
            }
            return true;
        }
    }

    // ALT allele frequency (diploid, dosage 0/1/2)
    let mut alt_freq = alt_sum / (2.0 * non_missing as f64);

    // Flip if ALT freq > 0.5
    if alt_freq > 0.5 {
        for g in row.iter_mut() {
            if *g >= 0.0 {
                *g = 2.0 - *g;
            }
        }
        std::mem::swap(ref_allele, alt_allele);
        // After flip, recompute alt_sum efficiently
        alt_sum = 2.0 * non_missing as f64 - alt_sum;
        alt_freq = alt_sum / (2.0 * non_missing as f64);
    }

    let maf = alt_freq.min(1.0 - alt_freq);
    if maf < maf_threshold as f64 {
        return false;
    }

    // Mean imputation for missing (do not round)
    let mean_g = alt_sum / non_missing as f64;
    let imputed: f32 = mean_g as f32;

    for g in row.iter_mut() {
        if *g < 0.0 {
            *g = imputed;
        }
    }

    true
}

// ============================================================
// BED chunk reader (SNP-major .bed)
// ============================================================

#[pyclass]
pub struct BedChunkReader {
    prefix: String,
    samples: Vec<String>,
    sites: Vec<SiteInfo>,
    mmap: Mmap,
    n_samples: usize,
    n_snps: usize,
    bytes_per_snp: usize,
    current_snp: usize,
    maf_threshold: f32,
    max_missing_rate: f32,
}

#[pymethods]
impl BedChunkReader {
    /// BedChunkReader(prefix, maf_threshold=None, max_missing_rate=None)
    ///
    /// Reads:
    ///   - {prefix}.bed/.bim/.fam
    ///
    /// Notes
    /// -----
    /// Only SNP-major BED is supported (header[2] must be 0x01).
    #[new]
    fn new(prefix: String, maf_threshold: Option<f32>, max_missing_rate: Option<f32>) -> PyResult<Self> {
        let samples = read_fam(&prefix).map_err(|msg| PyErr::new::<PyRuntimeError, _>(msg))?;
        let sites = read_bim(&prefix).map_err(|msg| PyErr::new::<PyRuntimeError, _>(msg))?;

        let n_samples = samples.len();
        let n_snps = sites.len();

        let bed_path = format!("{prefix}.bed");
        let file = File::open(&bed_path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        if mmap.len() < 3 {
            return Err(PyErr::new::<PyRuntimeError, _>("BED file too small (no header)"));
        }
        if mmap[0] != 0x6C || mmap[1] != 0x1B {
            return Err(PyErr::new::<PyRuntimeError, _>(
                "Malformed BED header (expected 0x6C 0x1B)",
            ));
        }
        if mmap[2] != 0x01 {
            return Err(PyErr::new::<PyRuntimeError, _>(
                "Only SNP-major BED is supported (header[2] must be 0x01)",
            ));
        }

        let bytes_per_snp = (n_samples + 3) / 4;

        Ok(Self {
            prefix,
            samples,
            sites,
            mmap,
            n_samples,
            n_snps,
            bytes_per_snp,
            current_snp: 0,
            maf_threshold: maf_threshold.unwrap_or(0.0),
            max_missing_rate: max_missing_rate.unwrap_or(1.0),
        })
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[getter]
    fn n_snps(&self) -> usize {
        self.n_snps
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> {
        self.samples.clone()
    }

    /// next_chunk(chunk_size) -> (geno, sites) or None
    ///
    /// Returns
    /// -------
    /// geno : numpy.ndarray[float32], shape (m_chunk, n_samples)
    /// sites: list[SiteInfo], length m_chunk
    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<Option<(&'py PyArray2<f32>, Vec<SiteInfo>)>> {
        if chunk_size == 0 {
            return Err(PyErr::new::<PyValueError, _>("chunk_size must be > 0"));
        }
        if self.current_snp >= self.n_snps {
            return Ok(None);
        }

        let mut geno_rows: Vec<Vec<f32>> = Vec::new();
        let mut out_sites: Vec<SiteInfo> = Vec::new();
        let data = &self.mmap[3..]; // skip BED header

        while self.current_snp < self.n_snps && geno_rows.len() < chunk_size {
            let snp_idx = self.current_snp;
            let offset = snp_idx * self.bytes_per_snp;
            let snp_bytes = &data[offset..offset + self.bytes_per_snp];

            // Decode one SNP into a float32 row (missing = -9.0)
            let mut row: Vec<f32> = vec![-9.0; self.n_samples];

            for (byte_idx, byte) in snp_bytes.iter().enumerate() {
                for within in 0..4 {
                    let samp_idx = byte_idx * 4 + within;
                    if samp_idx >= self.n_samples {
                        break;
                    }
                    let code = (byte >> (within * 2)) & 0b11;
                    // PLINK 2-bit coding (SNP-major):
                    // 00 -> homozygous A1 (0)
                    // 10 -> heterozygous (1)
                    // 11 -> homozygous A2 (2)
                    // 01 -> missing
                    let g = match code {
                        0b00 => 0.0_f32,
                        0b10 => 1.0_f32,
                        0b11 => 2.0_f32,
                        0b01 => -9.0_f32,
                        _ => -9.0_f32,
                    };
                    row[samp_idx] = g;
                }
            }

            let mut site = self.sites[snp_idx].clone();
            let keep = process_snp_row(
                &mut row,
                &mut site.ref_allele,
                &mut site.alt_allele,
                self.maf_threshold,
                self.max_missing_rate,
            );

            self.current_snp += 1;

            if keep {
                geno_rows.push(row);
                out_sites.push(site);
            }
        }

        if geno_rows.is_empty() {
            return Ok(None);
        }

        // Convert Vec<Vec<f32>> into ndarray::Array2 and return as numpy view
        let rows = geno_rows.len();
        let cols = self.n_samples;
        let mut mat = Array2::<f32>::zeros((rows, cols));
        for (i, row) in geno_rows.into_iter().enumerate() {
            for (j, g) in row.into_iter().enumerate() {
                mat[[i, j]] = g;
            }
        }

        #[allow(deprecated)]
        let py_mat = mat.into_pyarray(py);
        Ok(Some((py_mat, out_sites)))
    }
}

// ============================================================
// VCF chunk reader (VCF or VCF.GZ)
// ============================================================

fn open_text_maybe_gz(path: &Path) -> Result<Box<dyn BufRead + Send>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader: Box<dyn BufRead + Send> = if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    Ok(reader)
}

#[pyclass]
pub struct VcfChunkReader {
    samples: Vec<String>,
    reader: Box<dyn BufRead + Send>,
    finished: bool,
    maf_threshold: f32,
    max_missing_rate: f32,
}

#[pymethods]
impl VcfChunkReader {
    /// VcfChunkReader(path, maf_threshold=None, max_missing_rate=None)
    #[new]
    fn new(path: String, maf_threshold: Option<f32>, max_missing_rate: Option<f32>) -> PyResult<Self> {
        let p = Path::new(&path);
        let mut reader = open_text_maybe_gz(p).map_err(|msg| PyErr::new::<PyRuntimeError, _>(msg))?;

        // Parse VCF header and extract sample IDs from "#CHROM" line
        let mut header_line = String::new();
        let samples: Vec<String>;

        loop {
            header_line.clear();
            let n = reader
                .read_line(&mut header_line)
                .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

            if n == 0 {
                return Err(PyErr::new::<PyRuntimeError, _>("No #CHROM header found in VCF"));
            }

            if header_line.starts_with("#CHROM") {
                let parts: Vec<_> = header_line.trim_end().split('\t').collect();
                if parts.len() < 10 {
                    return Err(PyErr::new::<PyRuntimeError, _>("#CHROM header too short"));
                }
                samples = parts[9..].iter().map(|s| s.to_string()).collect();
                break;
            }
        }

        Ok(Self {
            samples,
            reader,
            finished: false,
            maf_threshold: maf_threshold.unwrap_or(0.0),
            max_missing_rate: max_missing_rate.unwrap_or(1.0),
        })
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> {
        self.samples.clone()
    }

    /// next_chunk(chunk_size) -> (geno, sites) or None
    ///
    /// Returns
    /// -------
    /// geno : numpy.ndarray[float32], shape (m_chunk, n_samples)
    /// sites: list[SiteInfo], length m_chunk
    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<Option<(&'py PyArray2<f32>, Vec<SiteInfo>)>> {
        if chunk_size == 0 {
            return Err(PyErr::new::<PyValueError, _>("chunk_size must be > 0"));
        }
        if self.finished {
            return Ok(None);
        }

        let mut geno_rows: Vec<Vec<f32>> = Vec::new();
        let mut sites: Vec<SiteInfo> = Vec::new();
        let mut line = String::new();

        while geno_rows.len() < chunk_size {
            line.clear();
            let n = self
                .reader
                .read_line(&mut line)
                .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

            if n == 0 {
                self.finished = true;
                break;
            }

            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }

            let parts: Vec<_> = line.trim_end().split('\t').collect();
            if parts.len() < 10 {
                continue;
            }

            let mut site = SiteInfo {
                chrom: parts[0].to_string(),
                pos: parts[1].parse().unwrap_or(0),
                ref_allele: parts[3].to_string(),
                alt_allele: parts[4].to_string(),
            };

            // Only decode GT; skip if GT is absent.
            let format = parts[8];
            if !format.split(':').any(|f| f == "GT") {
                continue;
            }

            let mut row: Vec<f32> = Vec::with_capacity(self.samples.len());
            for s in 9..parts.len() {
                let gt_field = parts[s];
                let gt = gt_field.split(':').next().unwrap_or(".");

                // Only supports biallelic 0/1 coding; others treated as missing.
                let g = match gt {
                    "0/0" | "0|0" => 0.0_f32,
                    "0/1" | "1/0" | "0|1" | "1|0" => 1.0_f32,
                    "1/1" | "1|1" => 2.0_f32,
                    "./." | ".|." => -9.0_f32,
                    _ => -9.0_f32,
                };
                row.push(g);
            }

            let keep = process_snp_row(
                &mut row,
                &mut site.ref_allele,
                &mut site.alt_allele,
                self.maf_threshold,
                self.max_missing_rate,
            );

            if keep {
                geno_rows.push(row);
                sites.push(site);
            }
        }

        if geno_rows.is_empty() {
            return Ok(None);
        }

        let rows = geno_rows.len();
        let cols = self.samples.len();
        let mut mat = Array2::<f32>::zeros((rows, cols));
        for (i, row) in geno_rows.into_iter().enumerate() {
            for (j, g) in row.into_iter().enumerate() {
                mat[[i, j]] = g;
            }
        }

        #[allow(deprecated)]
        let py_mat = mat.into_pyarray(py);
        Ok(Some((py_mat, sites)))
    }
}

// ============================================================
// Utility: count VCF SNP records (non-header lines)
// ============================================================

/// Count variant records (non-header lines) in a VCF/VCF.GZ file.
///
/// Header lines start with '#'. Variant records do not.
/// This performs a linear pass and is typically cheap compared to GWAS.
#[pyfunction]
fn count_vcf_snps(path: String) -> PyResult<usize> {
    let p = Path::new(&path);
    let mut reader = open_text_maybe_gz(p).map_err(|msg| PyErr::new::<PyRuntimeError, _>(msg))?;

    let mut n: usize = 0;
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        if bytes_read == 0 {
            break;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        n += 1;
    }

    Ok(n)
}

// ============================================================
// Streaming PLINK writer: write .fam once, stream .bim + .bed
// ============================================================

fn write_fam_simple(path: &Path, sample_ids: &[String], phenotype: Option<&[f64]>) -> Result<(), String> {
    let mut w = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
    for (i, sid) in sample_ids.iter().enumerate() {
        let ph = phenotype.map(|p| p[i]).unwrap_or(-9.0);
        // FID IID PID MID SEX PHENO
        writeln!(w, "{0}\t{0}\t0\t0\t1\t{1}", sid, ph).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[inline]
fn plink2bits_from_g_i8(g: i8) -> u8 {
    // PLINK bed 2-bit:
    // 00 = homozygous A1/A1
    // 10 = heterozygous
    // 11 = homozygous A2/A2
    // 01 = missing
    match g {
        0 => 0b00,
        1 => 0b10,
        2 => 0b11,
        _ => 0b01,
    }
}

#[pyclass]
pub struct PlinkStreamWriter {
    n_samples: usize,
    bed: BufWriter<File>,
    bim: BufWriter<File>,
    written_snps: usize,
}

#[pymethods]
impl PlinkStreamWriter {
    /// PlinkStreamWriter(prefix, sample_ids, phenotype=None)
    ///
    /// Creates:
    ///   - {prefix}.fam  (written once)
    ///   - {prefix}.bed  (written once with header, then streamed)
    ///   - {prefix}.bim  (streamed line-by-line)
    #[new]
    fn new(prefix: String, sample_ids: Vec<String>, phenotype: Option<Vec<f64>>) -> PyResult<Self> {
        if sample_ids.is_empty() {
            return Err(PyErr::new::<PyValueError, _>("sample_ids is empty"));
        }
        if let Some(ref p) = phenotype {
            if p.len() != sample_ids.len() {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "phenotype length mismatch: phenotype={}, n_samples={}",
                    p.len(),
                    sample_ids.len()
                )));
            }
        }

        let bed_path = format!("{prefix}.bed");
        let bim_path = format!("{prefix}.bim");
        let fam_path = format!("{prefix}.fam");

        // 1) write .fam once
        let ph_ref = phenotype.as_ref().map(|v| v.as_slice());
        write_fam_simple(Path::new(&fam_path), &sample_ids, ph_ref)
            .map_err(|e| PyErr::new::<PyIOError, _>(e))?;

        // 2) open .bed and write header once (SNP-major)
        let mut bed = BufWriter::new(
            File::create(&bed_path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?,
        );
        bed.write_all(&[0x6C, 0x1B, 0x01])
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        // 3) open .bim
        let bim = BufWriter::new(
            File::create(&bim_path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?,
        );

        Ok(Self {
            n_samples: sample_ids.len(),
            bed,
            bim,
            written_snps: 0,
        })
    }

    /// write_chunk(geno_chunk, sites)
    ///
    /// geno_chunk: ndarray[int8] shape (m_chunk, n_samples), SNP-major, -9 missing
    /// sites: Vec<SiteInfo> length m_chunk
    fn write_chunk(&mut self, geno_chunk: PyReadonlyArray2<i8>, sites: Vec<SiteInfo>) -> PyResult<()> {
        let arr = geno_chunk.as_array();
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(PyErr::new::<PyValueError, _>("geno_chunk must be 2D (m_chunk, n_samples)"));
        }

        let m_chunk = shape[0];
        let n_samples = shape[1];

        if n_samples != self.n_samples {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "n_samples mismatch: writer expects {}, got {}",
                self.n_samples, n_samples
            )));
        }
        if sites.len() != m_chunk {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "sites length mismatch: sites={}, m_chunk={}",
                sites.len(),
                m_chunk
            )));
        }

        // Write BIM lines for this chunk
        for s in sites.iter() {
            let snp_id = format!("{}_{}", s.chrom, s.pos);
            writeln!(
                self.bim,
                "{}\t{}\t0\t{}\t{}\t{}",
                s.chrom, snp_id, s.pos, s.ref_allele, s.alt_allele
            )
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        }

        // Write BED bytes SNP-by-SNP (packed 2-bit)
        let bytes_per_snp = (self.n_samples + 3) / 4;

        // ndarray strides are in number of elements
        let strides = arr.strides();
        let s0 = strides[0] as isize; // row stride
        let s1 = strides[1] as isize; // col stride
        let base = arr.as_ptr();      // *const i8

        unsafe {
            for snp in 0..m_chunk {
                let snp_off = (snp as isize) * s0;

                let mut i = 0usize;
                for _ in 0..bytes_per_snp {
                    let mut byte: u8 = 0;
                    for k in 0..4 {
                        let si = i + k;
                        let two = if si < self.n_samples {
                            let off = snp_off + (si as isize) * s1;
                            let g = *base.offset(off);
                            plink2bits_from_g_i8(g)
                        } else {
                            0b01 // padding missing
                        };
                        byte |= two << (k * 2);
                    }
                    self.bed
                        .write_all(&[byte])
                        .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
                    i += 4;
                }

                self.written_snps += 1;
            }
        }

        Ok(())
    }

    /// Flush buffered bytes to disk.
    fn flush(&mut self) -> PyResult<()> {
        self.bed.flush().map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        self.bim.flush().map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        Ok(())
    }

    /// Close writer (flush only; file handles will be dropped by Rust).
    fn close(&mut self) -> PyResult<()> {
        self.flush()?;
        Ok(())
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[getter]
    fn written_snps(&self) -> usize {
        self.written_snps
    }
}

// ============================================================
// Streaming VCF writer: plain .vcf OR gzip .vcf.gz (auto)
// ============================================================

#[inline]
fn vcf_gt_from_g_i8(g: i8) -> &'static str {
    match g {
        0 => "0/0",
        1 => "0/1",
        2 => "1/1",
        _ => "./.",
    }
}

/// Internal writer wrapper: either plain text or gzip-compressed text.
enum VcfOut {
    Plain(BufWriter<File>),
    Gzip(BufWriter<GzEncoder<File>>),
}

impl VcfOut {
    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match self {
            VcfOut::Plain(w) => w.write_all(buf),
            VcfOut::Gzip(w) => w.write_all(buf),
        }
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            VcfOut::Plain(w) => w.flush(),
            VcfOut::Gzip(w) => w.flush(),
        }
    }

    /// Finish the writer. For gzip, this is critical to write the gzip trailer (CRC/footer).
    #[inline]
    fn finish(mut self) -> std::io::Result<()> {
        self.flush()?;
        match self {
            VcfOut::Plain(_w) => Ok(()),
            VcfOut::Gzip(w) => {
                let enc = w.into_inner()?; // BufWriter -> GzEncoder
                let _file = enc.finish()?; // finalize gzip stream and return File
                Ok(())
            }
        }
    }
}

#[pyclass]
pub struct VcfStreamWriter {
    n_samples: usize,
    out: Option<VcfOut>, // Option allows take() on close
    written_snps: usize,
}

#[pymethods]
impl VcfStreamWriter {
    /// VcfStreamWriter(path, sample_ids)
    ///
    /// If `path` ends with ".gz", output is gzip-compressed (VCF.gz),
    /// otherwise it is a plain text VCF.
    ///
    /// This is "true streaming": variants are written chunk-by-chunk,
    /// without accumulating in memory.
    #[new]
    fn new(path: String, sample_ids: Vec<String>) -> PyResult<Self> {
        if sample_ids.is_empty() {
            return Err(PyErr::new::<PyValueError, _>("sample_ids is empty"));
        }

        let file = File::create(&path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        let mut out = if path.ends_with(".gz") {
            let enc = GzEncoder::new(file, Compression::default());
            VcfOut::Gzip(BufWriter::new(enc))
        } else {
            VcfOut::Plain(BufWriter::new(file))
        };

        // Write VCF headers once
        out.write_all(b"##fileformat=VCFv4.2\n")
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        out.write_all(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        // Write header row
        out.write_all(b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        for sid in sample_ids.iter() {
            out.write_all(b"\t")
                .and_then(|_| out.write_all(sid.as_bytes()))
                .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        }
        out.write_all(b"\n")
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

        Ok(Self {
            n_samples: sample_ids.len(),
            out: Some(out),
            written_snps: 0,
        })
    }

    /// write_chunk(geno_chunk, sites)
    ///
    /// geno_chunk: ndarray[int8] shape (m_chunk, n_samples), SNP-major, -9 missing
    /// sites: Vec[SiteInfo] length m_chunk
    fn write_chunk(&mut self, geno_chunk: PyReadonlyArray2<i8>, sites: Vec<SiteInfo>) -> PyResult<()> {
        let out = self
            .out
            .as_mut()
            .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("writer is closed"))?;

        let arr = geno_chunk.as_array();
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(PyErr::new::<PyValueError, _>("geno_chunk must be 2D (m_chunk, n_samples)"));
        }
        let m_chunk = shape[0];
        let n_samples = shape[1];

        if n_samples != self.n_samples {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "n_samples mismatch: writer expects {}, got {}",
                self.n_samples, n_samples
            )));
        }
        if sites.len() != m_chunk {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "sites length mismatch: sites={}, m_chunk={}",
                sites.len(),
                m_chunk
            )));
        }

        // ndarray strides in number of elements
        let strides = arr.strides();
        let s0 = strides[0] as isize; // row stride
        let s1 = strides[1] as isize; // col stride
        let base = arr.as_ptr();

        unsafe {
            for snp in 0..m_chunk {
                let s = &sites[snp];
                let snp_id = format!("{}_{}", s.chrom, s.pos);

                // Fixed 9 columns
                let prefix = format!(
                    "{}\t{}\t{}\t{}\t{}\t.\tPASS\t.\tGT",
                    s.chrom, s.pos, snp_id, s.ref_allele, s.alt_allele
                );
                out.write_all(prefix.as_bytes())
                    .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

                // Append sample GTs
                let snp_off = (snp as isize) * s0;
                for i in 0..self.n_samples {
                    let off = snp_off + (i as isize) * s1;
                    let g = *base.offset(off);
                    out.write_all(b"\t")
                        .and_then(|_| out.write_all(vcf_gt_from_g_i8(g).as_bytes()))
                        .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
                }

                out.write_all(b"\n")
                    .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

                self.written_snps += 1;
            }
        }

        Ok(())
    }

    /// Flush buffered bytes.
    /// For gzip output, this flushes encoder buffers but does not finalize gzip trailer.
    fn flush(&mut self) -> PyResult<()> {
        let out = self
            .out
            .as_mut()
            .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("writer is closed"))?;
        out.flush().map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        Ok(())
    }

    /// Close writer. For gzip output, this finalizes the gzip stream (writes trailer).
    fn close(&mut self) -> PyResult<()> {
        if let Some(out) = self.out.take() {
            out.finish().map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        }
        Ok(())
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[getter]
    fn written_snps(&self) -> usize {
        self.written_snps
    }
}

// ============================================================
// PyO3 module exports
// ============================================================

#[pymodule]
fn gfreader_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<SiteInfo>()?;
    m.add_class::<BedChunkReader>()?;
    m.add_class::<VcfChunkReader>()?;
    m.add_class::<PlinkStreamWriter>()?;
    m.add_class::<VcfStreamWriter>()?;
    m.add_function(wrap_pyfunction!(count_vcf_snps, m)?)?;
    Ok(())
}