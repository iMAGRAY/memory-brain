// Simple script to check ONNX Runtime version compatibility

fn main() {
    println!("Checking ONNX Runtime version...");
    
    // Set ORT_DYLIB_PATH environment variable
    std::env::set_var("ORT_DYLIB_PATH", "./onnxruntime/lib/onnxruntime.dll");
    
    // Try to initialize ONNX Runtime
    match ort::init() {
        Ok(_) => {
            println!("✓ ONNX Runtime initialized successfully!");
            
            // Try to get available providers
            match std::panic::catch_unwind(|| {
                // This will panic if version mismatch
                let _ = ort::session::builder::SessionBuilder::new();
            }) {
                Ok(_) => println!("✓ SessionBuilder created successfully!"),
                Err(e) => println!("✗ Failed to create SessionBuilder: {:?}", e),
            }
        }
        Err(e) => {
            println!("✗ Failed to initialize ONNX Runtime: {}", e);
        }
    }
}