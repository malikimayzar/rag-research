use pyo3::prelude::*;
use unicode_segmentation::UnicodeSegmentation;
use rayon::prelude::*;

#[pyfunction]
fn split_sentences_rs(text: &str) -> Vec<String> {
    // Memotong kalimat pake standar Unicode (jauh lebih bener & cepet dari Regex)
    text.unicode_sentences()
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() > 10)
        .collect()
}

#[pyfunction]
fn find_breakpoints_rs(similarities: Vec<f32>, threshold: f32) -> Vec<usize> {
    // Mencari titik potong berdasarkan similarity secara paralel
    similarities.par_iter()
        .enumerate()
        .filter(|(_, &sim)| sim < threshold)
        .map(|(i, _)| i + 1)
        .collect()
}

#[pyfunction]
fn assemble_chunks_rs(
    sentences: Vec<String>, 
    breakpoints: Vec<usize>, 
    overlap_size: usize
) -> Vec<Vec<String>> {
    let mut chunks = Vec::new();
    let mut prev_bp = 0;
    
    // Gabungin breakpoints dengan akhir kalimat
    let mut all_bps = breakpoints;
    all_bps.push(sentences.len());
    
    for (i, &bp) in all_bps.iter().enumerate() {
        let mut group: Vec<String> = sentences[prev_bp..bp].to_vec();
        
        // Logic Overlap: Ambil ekor dari chunk sebelumnya (kalo bukan chunk pertama)
        if i > 0 && overlap_size > 0 {
            let tail_start = if prev_bp >= overlap_size { prev_bp - overlap_size } else { 0 };
            let mut tail: Vec<String> = sentences[tail_start..prev_bp].to_vec();
            tail.append(&mut group);
            chunks.push(tail);
        } else {
            chunks.push(group);
        }
        
        prev_bp = bp;
    }
    chunks
}

/// Module Python yang akan di-export
#[pymodule]
fn semantic_chunker_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(split_sentences_rs, m)?)?;
    m.add_function(wrap_pyfunction!(find_breakpoints_rs, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_chunks_rs, m)?)?;
    Ok(())
}