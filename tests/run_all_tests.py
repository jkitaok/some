#!/usr/bin/env python3
"""
Test runner for all some package unit tests.

This script runs all unit tests in the tests directory and provides
a summary of the results.
"""
import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import some modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_and_run_tests():
    """Discover and run all tests in the tests directory."""
    # Get the directory containing this script
    test_dir = Path(__file__).parent
    
    # Discover all test files
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern='*_test.py')
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


def run_specific_test(test_module):
    """Run tests from a specific module."""
    try:
        # Import the test module
        module = __import__(f'{test_module}', fromlist=[''])
        
        # Load tests from the module
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except ImportError as e:
        print(f"Error importing test module '{test_module}': {e}")
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1]
        if not test_module.endswith('_test'):
            test_module += '_test'
        
        print(f"Running tests from {test_module}...")
        success = run_specific_test(test_module)
    else:
        # Run all tests
        print("Running all extraction package tests...")
        success = discover_and_run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
