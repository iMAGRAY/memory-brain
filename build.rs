use std::path::{Path, PathBuf};
use std::env;

// Constants for allowed library directories
#[cfg(not(target_os = "windows"))]
#[allow(dead_code)]
const UNIX_LIB_DIRS: &[&str] = &["/usr/lib", "/usr/local/lib", "/opt"];


fn main() {
    // Windows-specific linker flags to resolve ONNX Runtime conflicts
    #[cfg(all(target_os = "windows", target_env = "msvc", target_arch = "x86_64"))]
    {
        // Configure Windows C++ runtime linking for MSVC x64
        // This resolves conflicts between ONNX Runtime (MD_DynamicRelease) and esaxx-rs (MT_StaticRelease)
        // ONNX Runtime requires dynamic CRT, so we force all dependencies to use it
        println!("cargo:rustc-env=CRT_STATIC=false");  // Signal for dynamic CRT usage
        
        // Link to dynamic runtime libraries
        println!("cargo:rustc-link-arg=/DEFAULTLIB:msvcrt.lib");   // C runtime (dynamic)
        println!("cargo:rustc-link-arg=/DEFAULTLIB:msvcprt.lib");  // C++ runtime (dynamic) - resolves std::_Xlength_error
        
        // Explicitly exclude static runtime libraries to prevent conflicts
        println!("cargo:rustc-link-arg=/NODEFAULTLIB:libcmt.lib");   // Exclude static C runtime
        println!("cargo:rustc-link-arg=/NODEFAULTLIB:libcpmt.lib");  // Exclude static C++ runtime
        println!("cargo:rustc-link-arg=/NODEFAULTLIB:libcmtd.lib");  // Exclude debug static C runtime
        println!("cargo:rustc-link-arg=/NODEFAULTLIB:libcpmtd.lib"); // Exclude debug static C++ runtime
        
        // Add Windows API libraries required by vswhom-sys and ONNX Runtime
        // vswhom-sys needs these for Visual Studio detection and COM operations
        println!("cargo:rustc-link-lib=dylib=advapi32");  // Registry functions (RegOpenKeyExA, RegCloseKey, RegQueryValueExW)
        println!("cargo:rustc-link-lib=dylib=user32");    // Windows User API functions
        println!("cargo:rustc-link-lib=dylib=ole32");     // COM initialization and object management for VS detection
        println!("cargo:rustc-link-lib=dylib=oleaut32");  // OLE Automation for VARIANT and BSTR handling in VS COM
        
        // Find Visual Studio installation with validation
        let vc_path = env::var("VCINSTALLDIR")
            .ok()
            .filter(|p| Path::new(p).join("lib").exists())
            .unwrap_or_else(|| {
                let default_path = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC";
                if std::env::var("RUST_LOG").unwrap_or_default().contains("debug") {
                    eprintln!("[DEBUG] VCINSTALLDIR not found or invalid, using default: {}", default_path);
                }
                default_path.to_string()
            });
        
        // Add VC lib search path if it exists
        let vc_lib_path = format!("{}\\lib\\x64", vc_path);
        if Path::new(&vc_lib_path).exists() {
            println!("cargo:rustc-link-search=native={}", vc_lib_path);
            if std::env::var("RUST_LOG").unwrap_or_default().contains("debug") {
                eprintln!("[DEBUG] Added VC lib path: {}", vc_lib_path);
            }
        }
        
        if std::env::var("RUST_LOG").unwrap_or_default().contains("info") {
            eprintln!("[INFO] Applied CRT linking flags for Windows MSVC x64: dynamic CRT to resolve ONNX conflicts");
        }
        
        // Add search paths for ONNX Runtime if available with validation
        if let Ok(onnx_path) = std::env::var("ORT_LIB_LOCATION") {
            if validate_path(&onnx_path) {
                println!("cargo:rustc-link-search=native={}", onnx_path);
            } else {
                eprintln!("Warning: ORT_LIB_LOCATION contains invalid path: {}", onnx_path);
            }
        }
        
        // Create models directory for Windows (required for ONNX models)
        create_models_dir();
    }
}

/// Validates that the path is safe and exists within allowed base directories
#[allow(dead_code)]
fn validate_path(path: &str) -> bool {
    // Security: prevent path traversal attacks (including encoded variants)
    // For build scripts, environment variables should contain clean paths
    if path.contains("..") || path.contains("~") || 
       path.contains("%") || // Any percent-encoding should be rejected
       path.contains("\\\\") || path.contains("//") { // UNC paths or double slashes
        eprintln!("Warning: Path contains suspicious characters: {}", path);
        return false;
    }
    
    let path_obj = Path::new(path);
    
    // Only allow absolute paths that are within system library directories
    if !path_obj.is_absolute() {
        return false;
    }
    
    // Check if path exists before canonicalization
    if !path_obj.exists() {
        eprintln!("Warning: Path does not exist: {}", path);
        return false;
    }
    
    // Canonicalize to resolve any remaining path components and symlinks
    let normalized = match path_obj.canonicalize() {
        Ok(canonical) => canonical,
        Err(e) => {
            eprintln!("Warning: Failed to canonicalize path '{}': {}", path, e);
            return false;
        }
    };
    
    // Get allowed base directories for ONNX Runtime libraries
    let allowed_bases = get_allowed_lib_directories();
    
    // Verify the canonical path is within one of the allowed base directories
    let is_within_allowed_base = allowed_bases.iter().any(|base| {
        normalized.starts_with(base)
    });
    
    if !is_within_allowed_base {
        eprintln!("Warning: Path '{}' not within allowed base directories", normalized.display());
        return false;
    }
    
    // Final check: must be a directory and not a symlink to prevent symlink attacks
    normalized.is_dir() && !normalized.is_symlink()
}

/// Helper function to safely add and canonicalize a directory path
#[allow(dead_code)]
fn add_directory_if_valid(dirs: &mut Vec<PathBuf>, path: PathBuf, description: &str) {
    if let Ok(metadata) = path.metadata() {
        if metadata.is_dir() {
            match path.canonicalize() {
                Ok(canonical) => {
                    // Security: ensure the canonical path is not a symlink to prevent attacks
                    if canonical.is_symlink() {
                        eprintln!("Warning: Skipping {} directory '{}' - symlinks not allowed", 
                                 description, canonical.display());
                        return;
                    }
                    
                    // Additional security check: ensure path doesn't contain traversal sequences
                    let path_str = canonical.to_string_lossy();
                    if path_str.contains("..") {
                        eprintln!("Warning: Skipping {} directory '{}' - contains path traversal", 
                                 description, canonical.display());
                        return;
                    }
                    
                    // Verify directory still exists after canonicalization (prevents race conditions)
                    if !canonical.exists() || !canonical.is_dir() {
                        eprintln!("Warning: Skipping {} directory '{}' - no longer exists or not a directory", 
                                 description, canonical.display());
                        return;
                    }
                    
                    eprintln!("Added {} directory: {}", description, canonical.display());
                    dirs.push(canonical);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to canonicalize {} path '{}': {}", 
                             description, path.display(), e);
                }
            }
        }
    }
}

/// Returns allowed base directories for ONNX Runtime libraries
#[allow(dead_code)]
fn get_allowed_lib_directories() -> Vec<PathBuf> {
    let mut allowed_dirs = Vec::new();
    
    // Common Windows library locations
    #[cfg(target_os = "windows")]
    {
        // Program Files directories
        if let Ok(program_files) = env::var("ProgramFiles") {
            add_directory_if_valid(&mut allowed_dirs, PathBuf::from(program_files), "Program Files");
        }
        if let Ok(program_files_x86) = env::var("ProgramFiles(x86)") {
            add_directory_if_valid(&mut allowed_dirs, PathBuf::from(program_files_x86), "Program Files (x86)");
        }
        
        // Local AppData (for user installations)
        if let Ok(localappdata) = env::var("LOCALAPPDATA") {
            add_directory_if_valid(&mut allowed_dirs, PathBuf::from(localappdata), "Local AppData");
        }
        
        // System32 for system libraries
        if let Ok(systemroot) = env::var("SystemRoot") {
            add_directory_if_valid(&mut allowed_dirs, 
                                 PathBuf::from(systemroot).join("System32"), 
                                 "System32");
        }
        
        // Only allow specific safe development locations, not arbitrary current_dir
        // This prevents relative path manipulation if build happens in user-controlled dirs
    }
    
    // Unix-like systems library locations
    #[cfg(not(target_os = "windows"))]
    {
        // Add standard system library directories if they exist and are accessible
        for &lib_dir in UNIX_LIB_DIRS {
            let path = PathBuf::from(lib_dir);
            if let Ok(metadata) = path.metadata() {
                if metadata.is_dir() {
                    // Canonicalize to prevent symlink attacks
                    if let Ok(canonical) = path.canonicalize() {
                        eprintln!("Added Unix lib directory: {}", canonical.display());
                        allowed_dirs.push(canonical);
                    } else {
                        eprintln!("Warning: Failed to canonicalize Unix lib path: {}", path.display());
                    }
                }
            }
        }
        
        // Only allow specific safe subdirectories in HOME, not entire HOME directory
        // This prevents access to user files while allowing standard library locations
        if let Ok(home) = env::var("HOME") {
            let home_path = PathBuf::from(home);
            
            // .local/lib - Standard user library directory (follows XDG Base Directory)
            let local_lib = home_path.join(".local/lib");
            if let Ok(metadata) = local_lib.metadata() {
                if metadata.is_dir() {
                    if let Ok(canonical) = local_lib.canonicalize() {
                        eprintln!("Added user lib directory: {}", canonical.display());
                        allowed_dirs.push(canonical);
                    } else {
                        eprintln!("Warning: Failed to canonicalize user lib path: {}", local_lib.display());
                    }
                }
            }
            
            // .local/share - User data directory for applications
            let local_share = home_path.join(".local/share");
            if let Ok(metadata) = local_share.metadata() {
                if metadata.is_dir() {
                    if let Ok(canonical) = local_share.canonicalize() {
                        eprintln!("Added user share directory: {}", canonical.display());
                        allowed_dirs.push(canonical);
                    } else {
                        eprintln!("Warning: Failed to canonicalize user share path: {}", local_share.display());
                    }
                }
            }
        }
    }
    
    // Return directories that were successfully added and canonicalized
    allowed_dirs
}


/// Creates models directory safely with race condition protection
#[allow(dead_code)]
fn create_models_dir() {
    let models_dir = Path::new("models");
    
    // Check if directory already exists (fast path)
    if models_dir.exists() {
        return;
    }
    
    // Attempt to create directory
    match std::fs::create_dir_all(models_dir) {
        Ok(()) => {
            // Verify directory was actually created to prevent race conditions
            if !models_dir.exists() {
                eprintln!("Warning: Models directory creation reported success but directory does not exist");
            } else {
                eprintln!("Created models directory: {}", models_dir.display());
            }
        }
        Err(e) => {
            // Check if another process created it concurrently (race condition)
            if models_dir.exists() {
                eprintln!("Models directory created by another process during creation attempt");
            } else {
                eprintln!("Warning: Failed to create models directory: {}", e);
            }
        }
    }
    
    // Configure PyO3 for Python integration
    configure_pyo3();
}

/// Configure PyO3 Python integration
#[allow(dead_code)]
fn configure_pyo3() {
    // Tell cargo to rerun this build script if PyO3 environment variables change
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHONHOME");
    
    let _python_home = env::var("PYTHONHOME")
        .unwrap_or_else(|_| "C:\\Users\\1\\AppData\\Local\\Programs\\Python\\Python313".to_string());
    
    // Configure Windows Python linking
    #[cfg(target_os = "windows")]
    {
        let python_libs = format!("{}\\libs", python_home);
        let python_libs_path = Path::new(&python_libs);
        
        if python_libs_path.exists() {
            println!("cargo:rustc-link-search=native={}", python_libs);
            println!("cargo:rustc-link-lib=python313");
            
            // Also link python3.dll if available
            let python3_dll = format!("{}\\python3.dll", python_home);
            if Path::new(&python3_dll).exists() {
                println!("cargo:rustc-link-lib=dylib=python3");
            }
            
            println!("cargo:warning=PyO3 linked with Python at: {}", python_home);
        } else {
            println!("cargo:warning=Python libs not found at: {}", python_libs);
        }
    }
    
    // Ensure PyO3 uses the correct Python version
    if let Ok(pyo3_python) = env::var("PYO3_PYTHON") {
        println!("cargo:rustc-env=PYO3_PYTHON={}", pyo3_python);
        
        // Set Python version for PyO3
        if pyo3_python.contains("Python313") {
            println!("cargo:rustc-env=PYO3_PYTHON_VERSION=3.13");
        }
        
        println!("cargo:warning=Using PyO3 Python: {}", pyo3_python);
    }
}
