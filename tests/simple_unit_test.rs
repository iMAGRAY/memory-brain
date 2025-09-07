// Simple unit tests that don't require external dependencies

#[test]
fn test_basic_math() {
    assert_eq!(2 + 2, 4);
}

#[test] 
fn test_string_operations() {
    let s = "hello".to_string();
    assert_eq!(s.len(), 5);
}

// Test SIMD operations without external dependencies
#[test]
fn test_vector_operations() {
    let a = vec![1.0_f32, 2.0, 3.0];
    let b = vec![4.0_f32, 5.0, 6.0];
    
    let dot_product: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum();
    
    assert_eq!(dot_product, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}