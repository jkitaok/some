#!/usr/bin/env python3
"""
Test runner for all example tests.

This script runs all example tests and provides a summary of the results.
"""

import sys
import subprocess
from pathlib import Path

def run_test_file(test_file: Path, test_name: str) -> bool:
    """Run a single test file and return success status."""
    
    print(f"\n🧪 Running {test_name}")
    print("=" * 60)
    
    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=test_file.parent.parent.parent,  # Run from root directory
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Check if test passed
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED")
            return True
        else:
            print(f"❌ {test_name} FAILED (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} TIMED OUT")
        return False
    except Exception as e:
        print(f"💥 {test_name} ERROR: {e}")
        return False

def main():
    """Run all example tests."""
    
    print("🧪 Example Test Suite Runner")
    print("=" * 60)
    
    # Define all test files
    test_dir = Path(__file__).parent
    tests = [
        (test_dir / "test_dataset_standardization.py", "Dataset Standardization"),
        (test_dir / "test_io_media_integration.py", "IO & Media Integration"),
        (test_dir / "test_multimodal.py", "General Multimodal"),
        (test_dir / "vision_extraction" / "test_extraction.py", "Vision Extraction"),
        (test_dir / "audio_extraction" / "test_audio_extraction.py", "Audio Extraction"),
        (test_dir / "multimodal_extraction" / "test_multimodal.py", "Multimodal Extraction"),
    ]
    
    # Run each test
    results = []
    for test_file, test_name in tests:
        if test_file.exists():
            success = run_test_file(test_file, test_name)
            results.append((test_name, success))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EXAMPLE TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All example tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} example tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
