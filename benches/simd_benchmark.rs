use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn simd_benchmark(c: &mut Criterion) {
    c.bench_function("simd_operations", |b| {
        b.iter(|| {
            let data: Vec<f32> = (0..1000).map(|x| x as f32).collect();
            black_box(data.iter().sum::<f32>())
        })
    });
}

criterion_group\!(benches, simd_benchmark);
criterion_main\!(benches);
