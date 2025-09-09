#!/usr/bin/env python3
"""
Test script for verifying all installed Jupyter kernels
Version: 1.0.0
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


def run_command(cmd: List[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return -1, "", str(e)


def test_jupyter_installation() -> bool:
    """Test if Jupyter is properly installed."""
    print("Testing Jupyter installation...")

    commands = [
        (["jupyter", "--version"], "Jupyter version"),
        (["jupyter", "lab", "--version"], "JupyterLab version"),
        (["jupyter", "notebook", "--version"], "Jupyter Notebook version"),
    ]

    all_passed = True
    for cmd, description in commands:
        exit_code, stdout, stderr = run_command(cmd)
        if exit_code == 0:
            print(f"✅ {description}: {stdout.strip()}")
        else:
            print(f"❌ {description}: Failed - {stderr}")
            all_passed = False

    return all_passed


def list_installed_kernels() -> List[Dict[str, Any]]:
    """List all installed Jupyter kernels."""
    print("\nListing installed kernels...")

    exit_code, stdout, stderr = run_command(["jupyter", "kernelspec", "list", "--json"])

    if exit_code != 0:
        print(f"❌ Failed to list kernels: {stderr}")
        return []

    try:
        data = json.loads(stdout)
        kernels = []

        for name, spec in data.get("kernelspecs", {}).items():
            kernel_info = {
                "name": name,
                "display_name": spec.get("spec", {}).get("display_name", name),
                "language": spec.get("spec", {}).get("language", "unknown"),
                "path": spec.get("resource_dir", ""),
                "argv": spec.get("spec", {}).get("argv", []),
            }
            kernels.append(kernel_info)

            print(f"  • {kernel_info['display_name']} ({name})")
            print(f"    Language: {kernel_info['language']}")
            print(f"    Path: {kernel_info['path']}")

        return kernels
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse kernel list: {e}")
        return []


def test_kernel_execution(kernel_name: str, test_code: str) -> bool:
    """Test kernel execution with simple code."""
    print(f"\nTesting kernel '{kernel_name}'...")

    # Create a temporary notebook
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": test_code,
            }
        ],
        "metadata": {"kernelspec": {"name": kernel_name}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    # Save temporary notebook
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
        json.dump(notebook, f)
        notebook_path = f.name

    try:
        # Execute notebook
        exit_code, stdout, stderr = run_command(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=10",
                "--ExecutePreprocessor.kernel_name=" + kernel_name,
                "--inplace",
                notebook_path,
            ]
        )

        if exit_code == 0:
            print(f"✅ Kernel '{kernel_name}' executed successfully")
            return True
        else:
            print(f"❌ Kernel '{kernel_name}' execution failed: {stderr}")
            return False
    finally:
        # Clean up
        try:
            os.unlink(notebook_path)
        except:
            pass


def test_all_kernels():
    """Test all installed kernels."""
    print("=" * 60)
    print("Jupyter Kernel Test Suite")
    print("=" * 60)

    # Test Jupyter installation
    if not test_jupyter_installation():
        print("\n⚠️ Jupyter installation issues detected")
        return False

    # List kernels
    kernels = list_installed_kernels()
    if not kernels:
        print("\n⚠️ No kernels found")
        return False

    print(f"\nFound {len(kernels)} kernel(s)")

    # Define test code for each language
    test_codes = {
        "python": (
            "print('Hello from Python'); import sys; print(f'Python {sys.version}')"
        ),
        "rust": 'println!("Hello from Rust");',
        "bash": "echo 'Hello from Bash'; uname -a",
        "javascript": "console.log('Hello from JavaScript');",
        "typescript": "console.log('Hello from TypeScript');",
        "r": "print('Hello from R'); R.version.string",
    }

    # Test each kernel
    results = {}
    for kernel in kernels:
        language = kernel["language"].lower()
        test_code = test_codes.get(language, f"print('Testing {kernel['name']}')")

        success = test_kernel_execution(kernel["name"], test_code)
        results[kernel["name"]] = {
            "success": success,
            "language": language,
            "display_name": kernel["display_name"],
        }

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r["success"])
    failed = len(results) - passed

    print(f"\n✅ Passed: {passed}/{len(results)}")
    for name, result in results.items():
        if result["success"]:
            print(f"  • {result['display_name']} ({name})")

    if failed > 0:
        print(f"\n❌ Failed: {failed}/{len(results)}")
        for name, result in results.items():
            if not result["success"]:
                print(f"  • {result['display_name']} ({name})")

    # Overall result
    success_rate = (passed / len(results)) * 100 if results else 0
    print(f"\n📊 Success Rate: {success_rate:.1f}%")

    if success_rate >= 80:
        print("✅ Kernel test suite PASSED")
        return True
    else:
        print("❌ Kernel test suite FAILED")
        return False


def test_jupyter_api():
    """Test JupyterLab API endpoints."""
    print("\n" + "=" * 60)
    print("Testing JupyterLab API")
    print("=" * 60)

    import urllib.request
    import urllib.error

    base_url = "http://127.0.0.1:8888"
    endpoints = [
        ("/api", "API root"),
        ("/api/kernelspecs", "Kernel specifications"),
        ("/api/sessions", "Active sessions"),
        ("/api/terminals", "Terminal sessions"),
        ("/lab", "Lab interface"),
    ]

    results = []
    for endpoint, description in endpoints:
        url = base_url + endpoint
        try:
            response = urllib.request.urlopen(url, timeout=5)
            status = response.getcode()
            print(f"✅ {description} ({endpoint}): {status}")
            results.append(True)
        except urllib.error.HTTPError as e:
            print(f"❌ {description} ({endpoint}): HTTP {e.code}")
            results.append(False)
        except Exception as e:
            print(f"❌ {description} ({endpoint}): {e}")
            results.append(False)

    return all(results)


def main():
    """Main test execution."""
    print("Starting Jupyter Multi-Kernel Test Suite")
    print("Time:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print()

    # Test kernels
    kernel_test_passed = test_all_kernels()

    # Test API
    api_test_passed = test_jupyter_api()

    # Final result
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if kernel_test_passed and api_test_passed:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        if not kernel_test_passed:
            print("  • Kernel tests failed")
        if not api_test_passed:
            print("  • API tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
