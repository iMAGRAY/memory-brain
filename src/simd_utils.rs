//! SIMD utilities and CPU feature detection

use std::sync::OnceLock;

static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_sse: bool,
    pub has_sse2: bool,
    pub has_sse3: bool,
    pub has_ssse3: bool,
    pub has_sse41: bool,
    pub has_sse42: bool,
    pub has_avx: bool,
    pub has_fma: bool,
}

impl CpuFeatures {
    #[allow(dead_code)]
    const fn new() -> Self {
        CpuFeatures {
            has_avx2: false,
            has_sse: false,
            has_sse2: false,
            has_sse3: false,
            has_ssse3: false,
            has_sse41: false,
            has_sse42: false,
            has_avx: false,
            has_fma: false,
        }
    }
    
    /// Detect CPU features at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            CpuFeatures {
                has_avx2: is_x86_feature_detected!("avx2"),
                has_sse: is_x86_feature_detected!("sse"),
                has_sse2: is_x86_feature_detected!("sse2"),
                has_sse3: is_x86_feature_detected!("sse3"),
                has_ssse3: is_x86_feature_detected!("ssse3"),
                has_sse41: is_x86_feature_detected!("sse4.1"),
                has_sse42: is_x86_feature_detected!("sse4.2"),
                has_avx: is_x86_feature_detected!("avx"),
                has_fma: is_x86_feature_detected!("fma"),
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            CpuFeatures::new()
        }
    }
    
    /// Get cached CPU features (thread-safe)
    pub fn get() -> &'static CpuFeatures {
        CPU_FEATURES.get_or_init(|| Self::detect())
    }
    
    /// Get the best available SIMD level
    pub fn best_simd_level(&self) -> SimdLevel {
        if self.has_avx2 && self.has_fma {
            SimdLevel::Avx2Fma
        } else if self.has_avx2 {
            SimdLevel::Avx2
        } else if self.has_avx {
            SimdLevel::Avx
        } else if self.has_sse42 {
            SimdLevel::Sse42
        } else if self.has_sse2 {
            SimdLevel::Sse2
        } else {
            SimdLevel::Scalar
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    Scalar,
    Sse2,
    Sse42,
    Avx,
    Avx2,
    Avx2Fma,
}

impl SimdLevel {
    pub fn vector_width(&self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            SimdLevel::Sse2 | SimdLevel::Sse42 => 4,
            SimdLevel::Avx | SimdLevel::Avx2 | SimdLevel::Avx2Fma => 8,
        }
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            SimdLevel::Scalar => "Scalar",
            SimdLevel::Sse2 => "SSE2",
            SimdLevel::Sse42 => "SSE4.2",
            SimdLevel::Avx => "AVX",
            SimdLevel::Avx2 => "AVX2",
            SimdLevel::Avx2Fma => "AVX2+FMA",
        }
    }
}

/// Prefetch data into cache for better performance
/// 
/// On x86_64, uses _mm_prefetch to load data into L1 cache.
/// On other architectures, this is a no-op.
/// 
/// # Arguments
/// * `data` - Reference to data to prefetch
#[inline(always)]
pub fn prefetch_read<T>(data: &T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::_mm_prefetch;
        _mm_prefetch(data as *const T as *const i8, 0);
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = data; // Prevent unused variable warning
    }
}

/// Align vector length for optimal SIMD performance
/// 
/// Pads the vector with zeros to reach the next multiple of alignment.
/// This ensures SIMD operations can process full chunks without remainder.
/// 
/// # Arguments
/// * `v` - Vector to align
/// * `alignment` - Target alignment (typically 4 for SSE, 8 for AVX)
/// 
/// # Example
/// ```
/// let mut v = vec![1.0, 2.0, 3.0];
/// align_vector(&mut v, 4);
/// assert_eq!(v.len(), 4);
/// ```
pub fn align_vector(v: &mut Vec<f32>, alignment: usize) {
    if alignment == 0 {
        return;
    }
    let current_len = v.len();
    let aligned_len = (current_len + alignment - 1) / alignment * alignment;
    v.resize(aligned_len, 0.0);
}

/// Check if pointer is aligned to specified boundary
/// 
/// # Arguments
/// * `ptr` - Pointer to check
/// * `alignment` - Required alignment in bytes
/// 
/// # Returns
/// True if pointer is aligned, false otherwise
#[inline(always)]
pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
    alignment == 0 || (ptr as usize) % alignment == 0
}

/// Benchmark wrapper for SIMD operations
pub struct SimdBenchmark {
    iterations: usize,
    warmup_iterations: usize,
}

impl SimdBenchmark {
    pub fn new() -> Self {
        SimdBenchmark {
            iterations: 1000,
            warmup_iterations: 100,
        }
    }
    
    pub fn run<F: Fn()>(&self, name: &str, f: F) -> f64 {
        // Warmup
        for _ in 0..self.warmup_iterations {
            f();
        }
        
        // Actual benchmark
        let start = std::time::Instant::now();
        for _ in 0..self.iterations {
            f();
        }
        let duration = start.elapsed();
        
        let avg_time = duration.as_secs_f64() / self.iterations as f64;
        println!("{}: {:.3} μs per iteration", name, avg_time * 1_000_000.0);
        avg_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_features() {
        let features = CpuFeatures::detect();
        println!("CPU Features: {:?}", features);
        println!("Best SIMD Level: {:?}", features.best_simd_level());
    }
    
    #[test]
    fn test_alignment() {
        let mut v = vec![1.0, 2.0, 3.0];
        align_vector(&mut v, 8);
        assert_eq!(v.len(), 8);
        assert_eq!(v[3..], vec![0.0; 5]);
    }
}
