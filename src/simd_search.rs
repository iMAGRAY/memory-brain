//! SIMD-optimized vector search operations
//!
//! Provides high-performance cosine similarity calculations using SIMD instructions
//! for x86_64 and ARM architectures.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use std::cmp::Ordering;
use std::collections::HashMap;
use uuid::Uuid;

/// SIMD-optimized cosine similarity for x86_64 with AVX2
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    unsafe {
        let len = a.len();
        let simd_len = len - (len % 8);
        
        let mut dot_sum = _mm256_setzero_ps();
        let mut norm_a_sum = _mm256_setzero_ps();
        let mut norm_b_sum = _mm256_setzero_ps();
        
        // Process 8 elements at a time with AVX2
        for i in (0..simd_len).step_by(8) {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            
            // Dot product
            dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
            
            // Norms
            norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
            norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
        }
        
        // Sum the SIMD registers
        let dot_array: [f32; 8] = std::mem::transmute(dot_sum);
        let norm_a_array: [f32; 8] = std::mem::transmute(norm_a_sum);
        let norm_b_array: [f32; 8] = std::mem::transmute(norm_b_sum);
        
        let mut dot_product: f32 = dot_array.iter().sum();
        let mut norm_a_sq: f32 = norm_a_array.iter().sum();
        let mut norm_b_sq: f32 = norm_b_array.iter().sum();
        
        // Process remaining elements
        for i in simd_len..len {
            dot_product += a[i] * b[i];
            norm_a_sq += a[i] * a[i];
            norm_b_sq += b[i] * b[i];
        }
        
        let norm_a = norm_a_sq.sqrt();
        let norm_b = norm_b_sq.sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// SIMD-optimized cosine similarity for x86_64 with SSE
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    unsafe {
        let len = a.len();
        let simd_len = len - (len % 4);
        
        let mut dot_sum = _mm_setzero_ps();
        let mut norm_a_sum = _mm_setzero_ps();
        let mut norm_b_sum = _mm_setzero_ps();
        
        // Process 4 elements at a time with SSE
        for i in (0..simd_len).step_by(4) {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            
            // Dot product
            let prod = _mm_mul_ps(va, vb);
            dot_sum = _mm_add_ps(dot_sum, prod);
            
            // Norms
            let va_sq = _mm_mul_ps(va, va);
            let vb_sq = _mm_mul_ps(vb, vb);
            norm_a_sum = _mm_add_ps(norm_a_sum, va_sq);
            norm_b_sum = _mm_add_ps(norm_b_sum, vb_sq);
        }
        
        // Sum the SIMD registers
        let dot_array: [f32; 4] = std::mem::transmute(dot_sum);
        let norm_a_array: [f32; 4] = std::mem::transmute(norm_a_sum);
        let norm_b_array: [f32; 4] = std::mem::transmute(norm_b_sum);
        
        let mut dot_product: f32 = dot_array.iter().sum();
        let mut norm_a_sq: f32 = norm_a_array.iter().sum();
        let mut norm_b_sq: f32 = norm_b_array.iter().sum();
        
        // Process remaining elements
        for i in simd_len..len {
            dot_product += a[i] * b[i];
            norm_a_sq += a[i] * a[i];
            norm_b_sq += b[i] * b[i];
        }
        
        let norm_a = norm_a_sq.sqrt();
        let norm_b = norm_b_sq.sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// SIMD-optimized cosine similarity for ARM NEON
#[cfg(target_arch = "aarch64")]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    unsafe {
        let len = a.len();
        let simd_len = len - (len % 4);
        
        let mut dot_sum = vdupq_n_f32(0.0);
        let mut norm_a_sum = vdupq_n_f32(0.0);
        let mut norm_b_sum = vdupq_n_f32(0.0);
        
        // Process 4 elements at a time with NEON
        for i in (0..simd_len).step_by(4) {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            
            // Dot product
            dot_sum = vfmaq_f32(dot_sum, va, vb);
            
            // Norms
            norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
            norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);
        }
        
        // Sum the SIMD registers
        let dot_product_partial = vaddvq_f32(dot_sum);
        let norm_a_partial = vaddvq_f32(norm_a_sum);
        let norm_b_partial = vaddvq_f32(norm_b_sum);
        
        let mut dot_product = dot_product_partial;
        let mut norm_a_sq = norm_a_partial;
        let mut norm_b_sq = norm_b_partial;
        
        // Process remaining elements
        for i in simd_len..len {
            dot_product += a[i] * b[i];
            norm_a_sq += a[i] * a[i];
            norm_b_sq += b[i] * b[i];
        }
        
        let norm_a = norm_a_sq.sqrt();
        let norm_b = norm_b_sq.sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// Fallback implementation for platforms without SIMD
#[cfg(not(any(
    all(target_arch = "x86_64"),
    target_arch = "aarch64"
)))]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    cosine_similarity_scalar(a, b)
}

/// Scalar fallback implementation
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}


/// Optimized vector search for HashMap-based embeddings
pub fn parallel_vector_search(
    embeddings: &HashMap<Uuid, Vec<f32>>, 
    query: &[f32], 
    top_k: usize
) -> Result<Vec<SimilarityMatch>, String> {
    use rayon::prelude::*;
    
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }
    
    let mut similarities: Vec<SimilarityMatch> = embeddings
        .par_iter()
        .map(|(id, embedding)| {
            let similarity = cosine_similarity_simd(query, embedding);
            SimilarityMatch {
                memory_id: *id,
                similarity,
                context_path: String::new(), // Will be filled by caller if needed
            }
        })
        .collect();
    
    // Early return if no similarities found
    if similarities.is_empty() {
        return Ok(Vec::new());
    }
    
    // Efficient top-k selection
    let k = top_k.min(similarities.len());
    if k == 0 {
        return Ok(Vec::new());
    }
    
    // Partial sort for top-k (more efficient than full sort)
    let nth_index = k.saturating_sub(1);
    similarities.select_nth_unstable_by(nth_index, |a, b| {
        b.similarity.partial_cmp(&a.similarity).unwrap_or(Ordering::Equal)
    });
    
    similarities.truncate(k);
    similarities.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(Ordering::Equal));
    
    Ok(similarities)
}

/// Similarity match result for vector search
#[derive(Debug, Clone)]
pub struct SimilarityMatch {
    pub memory_id: Uuid,
    pub similarity: f32,
    pub context_path: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_simd() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity_simd(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity_simd(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity_simd(&a, &d) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_simd_vs_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        
        let simd_result = cosine_similarity_simd(&a, &b);
        let scalar_result = cosine_similarity_scalar(&a, &b);
        
        assert!((simd_result - scalar_result).abs() < 0.0001);
    }

    #[test]
    fn test_parallel_search() {
        use uuid::Uuid;
        use std::collections::HashMap;
        
        let query = vec![1.0, 0.0, 0.0];
        let mut embeddings = HashMap::new();
        
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let id4 = Uuid::new_v4();
        
        embeddings.insert(id1, vec![1.0, 0.0, 0.0]);  // Perfect match
        embeddings.insert(id2, vec![0.0, 1.0, 0.0]);  // Orthogonal
        embeddings.insert(id3, vec![0.7, 0.7, 0.0]);  // Similar
        embeddings.insert(id4, vec![-1.0, 0.0, 0.0]); // Opposite
        
        let results = parallel_vector_search(&embeddings, &query, 2).unwrap();
        
        assert_eq!(results.len(), 2);
        assert!(results[0].similarity > results[1].similarity); // Check ordering
        
        // The first result should be the perfect match
        assert_eq!(results[0].memory_id, id1);
        assert!((results[0].similarity - 1.0).abs() < 0.001);
    }
}