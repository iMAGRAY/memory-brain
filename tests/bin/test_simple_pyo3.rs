// Simple PyO3 test - verify Python integration works
use pyo3::prelude::*;
use std::time::Instant;

fn main() -> PyResult<()> {
    println!("🐍 PyO3 Integration Test");
    println!("========================\n");

    // Test 1: Python initialization
    println!("1. Testing Python initialization...");
    let start = Instant::now();
    
    Python::with_gil(|py| -> PyResult<()> {
        println!("  ✅ Python GIL acquired");
        println!("     Python version: {}", py.version());
        println!("     Time: {:?}\n", start.elapsed());

        // Test 2: Import basic modules
        println!("2. Testing module imports...");
        let start = Instant::now();
        
        let sys = py.import_bound("sys")?;
        let platform = py.import_bound("platform")?;
        
        let python_version: String = sys.getattr("version")?.extract()?;
        let platform_info: String = platform.call_method0("platform")?.extract()?;
        
        println!("  ✅ Basic modules imported");
        println!("     Full version: {}", python_version.lines().next().unwrap_or("unknown"));
        println!("     Platform: {}", platform_info);
        println!("     Time: {:?}\n", start.elapsed());

        // Test 3: NumPy availability (critical for embeddings)
        println!("3. Testing NumPy availability...");
        let start = Instant::now();
        
        match py.import_bound("numpy") {
            Ok(numpy) => {
                let numpy_version: String = numpy.getattr("__version__")?.extract()?;
                println!("  ✅ NumPy imported successfully");
                println!("     NumPy version: {}", numpy_version);
                
                // Test basic numpy array creation
                let array = numpy.call_method1("array", ([1, 2, 3, 4, 5],))?;
                let array_shape: (usize,) = array.getattr("shape")?.extract()?;
                println!("     Test array shape: {:?}", array_shape);
            }
            Err(e) => {
                println!("  ❌ NumPy not available: {}", e);
                println!("     This is critical for embedding operations");
            }
        }
        println!("     Time: {:?}\n", start.elapsed());

        // Test 4: Sentence Transformers availability
        println!("4. Testing Sentence Transformers...");
        let start = Instant::now();
        
        match py.import_bound("sentence_transformers") {
            Ok(st) => {
                let st_version: PyResult<String> = st.getattr("__version__")?.extract();
                match st_version {
                    Ok(version) => {
                        println!("  ✅ Sentence Transformers available");
                        println!("     Version: {}", version);
                    }
                    Err(_) => {
                        println!("  ✅ Sentence Transformers module found (version not accessible)");
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Sentence Transformers not available: {}", e);
                println!("     This is required for text embedding generation");
                println!("     Install with: pip install sentence-transformers");
            }
        }
        println!("     Time: {:?}\n", start.elapsed());

        // Test 5: Torch availability (backend for sentence transformers)
        println!("5. Testing PyTorch...");
        let start = Instant::now();
        
        match py.import_bound("torch") {
            Ok(torch) => {
                let torch_version: String = torch.getattr("__version__")?.extract()?;
                println!("  ✅ PyTorch available");
                println!("     Version: {}", torch_version);
                
                // Check CUDA availability safely
                let cuda_module = torch.getattr("cuda")?;
                let cuda_available: bool = cuda_module.call_method0("is_available")?.extract()?;
                if cuda_available {
                    let cuda_device_count: i32 = cuda_module.call_method0("device_count")?.extract()?;
                    println!("     🚀 CUDA available with {} devices", cuda_device_count);
                } else {
                    println!("     ⚠️  CUDA not available - using CPU");
                }
            }
            Err(e) => {
                println!("  ❌ PyTorch not available: {}", e);
                println!("     This is the backend for Sentence Transformers");
            }
        }
        println!("     Time: {:?}\n", start.elapsed());

        // Test 6: Safe computation test using built-in modules
        println!("6. Testing Python computation...");
        let start = Instant::now();
        
        // Safe computation using only standard library functions
        let math_module = py.import_bound("math")?;
        let sin_func = math_module.getattr("sin")?;
        
        // Test computation with known safe values
        let test_values = [0.0, 0.1, 0.2, 0.3, 0.4];
        let mut computed_count = 0;
        
        for &value in &test_values {
            let _result: f64 = sin_func.call1((value,))?.extract()?;
            computed_count += 1;
        }
        
        println!("  ✅ Python computation completed");
        println!("     Computed {} sine values safely", computed_count);
        println!("     Time: {:?}\n", start.elapsed());

        Ok(())
    })?;

    // Summary
    println!("🎯 PyO3 Integration Summary");
    println!("===========================");
    println!("✅ Python GIL: Working");
    println!("✅ Basic modules: Importable");
    
    Python::with_gil(|py| {
        match py.import_bound("numpy") {
            Ok(_) => println!("✅ NumPy: Available"),
            Err(_) => println!("❌ NumPy: Missing"),
        }
        
        match py.import_bound("sentence_transformers") {
            Ok(_) => println!("✅ Sentence Transformers: Available"),
            Err(_) => println!("❌ Sentence Transformers: Missing"),
        }
        
        match py.import_bound("torch") {
            Ok(_) => println!("✅ PyTorch: Available"),
            Err(_) => println!("❌ PyTorch: Missing"),
        }
    });

    println!("\nNext steps for full integration:");
    println!("1. If any packages are missing, install them:");
    println!("   pip install torch sentence-transformers numpy");
    println!("2. Test actual embedding generation");
    println!("3. Integrate with Rust memory service");

    Ok(())
}