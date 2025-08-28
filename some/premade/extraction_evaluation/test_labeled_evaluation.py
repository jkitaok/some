"""
Test script for labeled data evaluation functionality.

This script tests the labeled data evaluation system without requiring API calls.
It validates data loading, formatting, and utility functions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from .labeled_utils import (
    load_labeled_data,
    validate_labeled_item,
    format_evaluation_record,
    calculate_accuracy_metrics,
    analyze_field_differences,
    calculate_field_accuracy
)


def test_data_loading():
    """Test loading and validation of labeled data."""
    print("🧪 Testing Data Loading")
    print("=" * 40)
    
    # Test with sample data
    current_dir = Path(__file__).parent
    sample_data_path = current_dir / "sample_labeled_data.json"
    
    try:
        labeled_data = load_labeled_data(str(sample_data_path))
        print(f"✅ Successfully loaded {len(labeled_data)} items")
        
        # Validate each item
        for i, item in enumerate(labeled_data):
            validated_item = validate_labeled_item(item)
            print(f"✅ Item {i+1} ({validated_item['id']}) validated successfully")
            
        return labeled_data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def test_accuracy_calculations():
    """Test accuracy calculation functions."""
    print("\n🧪 Testing Accuracy Calculations")
    print("=" * 40)
    
    # Test data
    ground_truth = {
        "name": "iPhone 15 Pro",
        "price": 999.0,
        "features": ["titanium design", "A17 Pro chip", "improved camera"],
        "colors": ["Natural Titanium", "Blue Titanium"]
    }
    
    # Perfect match
    perfect_extraction = ground_truth.copy()
    perfect_metrics = calculate_accuracy_metrics(perfect_extraction, ground_truth)
    print(f"✅ Perfect match accuracy: {perfect_metrics['accuracy_score']:.2f}")
    
    # Partial match
    partial_extraction = {
        "name": "iPhone 15 Pro",
        "price": 999.0,
        "features": ["titanium design", "A17 Pro chip"],  # Missing one feature
        "colors": ["Natural Titanium", "Blue Titanium", "Extra Color"]  # Extra color
    }
    partial_metrics = calculate_accuracy_metrics(partial_extraction, ground_truth)
    print(f"✅ Partial match accuracy: {partial_metrics['accuracy_score']:.2f}")
    
    # Field accuracy test
    field_accuracy = calculate_field_accuracy(partial_extraction, ground_truth)
    print(f"✅ Field accuracy: {field_accuracy}")
    
    # Field differences test
    differences = analyze_field_differences(partial_extraction, ground_truth)
    print(f"✅ Field differences: {differences}")
    
    return True


def test_evaluation_record_formatting():
    """Test formatting of evaluation records."""
    print("\n🧪 Testing Evaluation Record Formatting")
    print("=" * 40)
    
    # Sample labeled item
    labeled_item = {
        "id": "test_001",
        "original_text": "Test product description",
        "ground_truth": {"name": "Test Product", "price": 99.99},
        "extraction_prompt": "Extract product info",
        "expected_schema": {"name": "string", "price": "number"}
    }
    
    # Sample extraction output
    extraction_output = {"name": "Test Product", "price": 99.99}
    
    try:
        formatted_record = format_evaluation_record(labeled_item, extraction_output)
        
        required_fields = ["original_text", "extraction_prompt", "expected_schema", 
                          "extraction_output", "ground_truth", "evaluation_context"]
        
        for field in required_fields:
            if field not in formatted_record:
                print(f"❌ Missing field: {field}")
                return False
        
        print("✅ Evaluation record formatted successfully")
        print(f"   Fields: {list(formatted_record.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Error formatting record: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases")
    print("=" * 40)
    
    # Empty extraction vs ground truth
    empty_extraction = {}
    ground_truth = {"name": "Product", "price": 99.99}
    
    metrics = calculate_accuracy_metrics(empty_extraction, ground_truth)
    print(f"✅ Empty extraction accuracy: {metrics['accuracy_score']:.2f}")
    
    differences = analyze_field_differences(empty_extraction, ground_truth)
    print(f"✅ Missing fields detected: {differences['missing_fields']}")
    
    # Extra fields in extraction
    extra_extraction = {"name": "Product", "price": 99.99, "extra_field": "extra"}
    differences = analyze_field_differences(extra_extraction, ground_truth)
    print(f"✅ Extra fields detected: {differences['extra_fields']}")
    
    # List comparison edge cases
    gt_with_lists = {"tags": ["tag1", "tag2", "tag3"]}
    ext_with_lists = {"tags": ["tag1", "tag2"]}  # Missing one tag
    
    list_metrics = calculate_accuracy_metrics(ext_with_lists, gt_with_lists)
    print(f"✅ List field metrics - Precision: {list_metrics['precision']:.2f}, Recall: {list_metrics['recall']:.2f}")
    
    return True


def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("🚀 Running Comprehensive Labeled Data Evaluation Tests")
    print("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Accuracy Calculations", test_accuracy_calculations),
        ("Record Formatting", test_evaluation_record_formatting),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result is not False
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 40)
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Labeled data evaluation system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
