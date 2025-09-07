// Simple SIMD test binary without PyO3 dependencies
use ai_memory_service::simd_search::cosine_similarity_simd;

fn main() {
    println!("Testing SIMD functions without PyO3...\n");
    
    // Test 1: Identical vectors
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let similarity = cosine_similarity_simd(&a, &b);
    println!("Test 1 - Identical vectors: {:.4} (expected: 1.0)", similarity);
    assert!((similarity - 1.0).abs() < 0.001);
    
    // Test 2: Orthogonal vectors
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let similarity = cosine_similarity_simd(&a, &b);
    println!("Test 2 - Orthogonal vectors: {:.4} (expected: 0.0)", similarity);
    assert!(similarity.abs() < 0.001);
    
    // Test 3: Opposite vectors
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![-1.0, 0.0, 0.0];
    let similarity = cosine_similarity_simd(&a, &b);
    println!("Test 3 - Opposite vectors: {:.4} (expected: -1.0)", similarity);
    assert!((similarity + 1.0).abs() < 0.001);
    
    // Test 4: Normalized vectors
    let a = vec![0.6, 0.8];
    let b = vec![0.8, 0.6];
    let similarity = cosine_similarity_simd(&a, &b);
    println!("Test 4 - Similar vectors: {:.4} (expected: ~0.96)", similarity);
    assert!((similarity - 0.96).abs() < 0.01);
    
    // Test 5: Large vectors (test SIMD performance)
    let size = 1024;
    let a: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..size).map(|i| (i as f32).cos()).collect();
    let similarity = cosine_similarity_simd(&a, &b);
    println!("Test 5 - Large vectors (1024 dims): {:.4}", similarity);
    
    println!("\n✅ All SIMD tests passed!");
    
    // Check CPU features
    println!("\nCPU Features:");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            println!("  ✓ AVX2 supported");
        } else {
            println!("  ✗ AVX2 not supported");
        }
        
        if is_x86_feature_detected!("sse") {
            println!("  ✓ SSE supported");
        } else {
            println!("  ✗ SSE not supported");
        }
        
        if is_x86_feature_detected!("sse2") {
            println!("  ✓ SSE2 supported");
        } else {
            println!("  ✗ SSE2 not supported");
        }
        
        if is_x86_feature_detected!("sse3") {
            println!("  ✓ SSE3 supported");
        } else {
            println!("  ✗ SSE3 not supported");
        }
        
        if is_x86_feature_detected!("ssse3") {
            println!("  ✓ SSSE3 supported");
        } else {
            println!("  ✗ SSSE3 not supported");
        }
        
        if is_x86_feature_detected!("sse4.1") {
            println!("  ✓ SSE4.1 supported");
        } else {
            println!("  ✗ SSE4.1 not supported");
        }
        
        if is_x86_feature_detected!("sse4.2") {
            println!("  ✓ SSE4.2 supported");
        } else {
            println!("  ✗ SSE4.2 not supported");
        }
        
        if is_x86_feature_detected!("avx") {
            println!("  ✓ AVX supported");
        } else {
            println!("  ✗ AVX not supported");
        }
    }
}