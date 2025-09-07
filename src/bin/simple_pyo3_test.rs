use pyo3::prelude::*;

#[tokio::main]
async fn main() -> PyResult<()> {
    println!("Testing PyO3 basic functionality...");
    
    Python::with_gil(|py| {
        println!("✓ Python GIL acquired successfully!");
        
        // Test basic Python operations
        let result: i32 = py.eval_bound("2 + 3", None, None)?.extract()?;
        println!("✓ Python eval (2+3): {}", result);
        
        // Test importing sys
        let sys = py.import_bound("sys")?;
        let version: String = sys.getattr("version")?.extract()?;
        println!("✓ Python version: {}", version.lines().next().unwrap_or("unknown"));
        
        // Test importing torch
        match py.import_bound("torch") {
            Ok(_) => println!("✓ torch imported successfully"),
            Err(e) => println!("⚠ Failed to import torch: {}", e),
        }
        
        // Test importing sentence_transformers
        match py.import_bound("sentence_transformers") {
            Ok(_) => println!("✓ sentence_transformers imported successfully"),
            Err(e) => println!("⚠ Failed to import sentence_transformers: {}", e),
        }
        
        println!("🎉 PyO3 integration working!");
        Ok(())
    })
}