@echo off
echo Setting up Python environment...
set PYTHONHOME=C:\Users\1\AppData\Local\Programs\Python\Python313
set PATH=C:\Users\1\AppData\Local\Programs\Python\Python313;C:\Users\1\AppData\Local\Programs\Python\Python313\Scripts;%PATH%

echo Running integration tests...
cargo test --test integration_test -- --nocapture