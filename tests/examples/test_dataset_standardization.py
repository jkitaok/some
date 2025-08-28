#!/usr/bin/env python3
"""
Test script to verify that all input datasets follow the standard format.

Standard format requires:
- id: str - Unique identifier
- text: str - Text content (can be null/empty)
- image_path: str|null - Path to image file
- audio_path: str|null - Path to audio file

This script validates all example datasets against this standard.
"""

import sys
from pathlib import Path

# Add the root directory to the path so we can import from some
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from some.io import read_json

def validate_dataset_format(dataset_path: Path, dataset_name: str) -> bool:
    """Validate that a dataset follows the standard format."""
    
    print(f"\n🔍 Validating {dataset_name}")
    print("=" * 50)
    
    # Check if file exists
    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    # Load dataset
    data = read_json(str(dataset_path))
    if data is None:
        print(f"❌ Failed to load dataset: {dataset_path}")
        return False
    
    if not isinstance(data, list):
        print(f"❌ Dataset must be a list, got {type(data)}")
        return False
    
    print(f"📊 Found {len(data)} items in dataset")
    
    # Required fields
    required_fields = ["id", "text", "image_path", "audio_path"]
    
    # Validate each item
    valid_items = 0
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"❌ Item {i} is not a dictionary")
            continue
        
        # Check required fields
        missing_fields = []
        for field in required_fields:
            if field not in item:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Item {i} ({item.get('id', 'unknown')}) missing fields: {missing_fields}")
            continue
        
        # Validate field types
        field_errors = []
        
        # id must be string
        if not isinstance(item["id"], str) or not item["id"].strip():
            field_errors.append("id must be non-empty string")
        
        # text must be string or null
        if item["text"] is not None and not isinstance(item["text"], str):
            field_errors.append("text must be string or null")
        
        # image_path must be string or null
        if item["image_path"] is not None and not isinstance(item["image_path"], str):
            field_errors.append("image_path must be string or null")
        
        # audio_path must be string or null
        if item["audio_path"] is not None and not isinstance(item["audio_path"], str):
            field_errors.append("audio_path must be string or null")
        
        if field_errors:
            print(f"❌ Item {i} ({item['id']}) field errors: {field_errors}")
            continue
        
        valid_items += 1
    
    success_rate = valid_items / len(data) * 100 if data else 0
    print(f"✅ Valid items: {valid_items}/{len(data)} ({success_rate:.1f}%)")
    
    if valid_items == len(data):
        print(f"🎉 {dataset_name} fully compliant with standard format!")
        return True
    else:
        print(f"⚠️  {dataset_name} has format issues")
        return False

def analyze_dataset_coverage(dataset_path: Path, dataset_name: str) -> dict:
    """Analyze what modalities are covered in the dataset."""
    
    data = read_json(str(dataset_path))
    if not data:
        return {}
    
    coverage = {
        "text_only": 0,
        "image_only": 0,
        "audio_only": 0,
        "text_image": 0,
        "text_audio": 0,
        "image_audio": 0,
        "text_image_audio": 0,
        "empty": 0
    }
    
    for item in data:
        has_text = bool(item.get("text", "").strip())
        has_image = bool(item.get("image_path"))
        has_audio = bool(item.get("audio_path"))
        
        if has_text and has_image and has_audio:
            coverage["text_image_audio"] += 1
        elif has_text and has_image:
            coverage["text_image"] += 1
        elif has_text and has_audio:
            coverage["text_audio"] += 1
        elif has_image and has_audio:
            coverage["image_audio"] += 1
        elif has_text:
            coverage["text_only"] += 1
        elif has_image:
            coverage["image_only"] += 1
        elif has_audio:
            coverage["audio_only"] += 1
        else:
            coverage["empty"] += 1
    
    print(f"\n📊 {dataset_name} Modality Coverage:")
    for modality, count in coverage.items():
        if count > 0:
            print(f"  {modality.replace('_', ' + ').title()}: {count} items")
    
    return coverage

def main():
    """Test all example datasets for standard format compliance."""
    
    print("🧪 Dataset Standardization Test Suite")
    print("=" * 60)
    
    # Define datasets to test
    root_dir = Path(__file__).parent.parent.parent
    examples_dir = root_dir / "some" / "examples"
    datasets = [
        (examples_dir / "vision_extraction" / "input_dataset" / "sample_products.json", "Vision Extraction"),
        (examples_dir / "audio_extraction" / "input_dataset" / "sample_audio.json", "Audio Extraction"),
        (examples_dir / "multimodal_extraction" / "input_dataset" / "sample_multimodal.json", "Multimodal Extraction"),
        (examples_dir / "multimodal_extraction" / "input_dataset" / "test_samples.json", "Multimodal Test Samples")
    ]
    
    # Test each dataset
    results = []
    for dataset_path, dataset_name in datasets:
        is_valid = validate_dataset_format(dataset_path, dataset_name)
        results.append((dataset_name, is_valid))
        
        if is_valid:
            analyze_dataset_coverage(dataset_path, dataset_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 STANDARDIZATION RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, valid in results if valid)
    total = len(results)
    
    for dataset_name, is_valid in results:
        status = "✅ COMPLIANT" if is_valid else "❌ NON-COMPLIANT"
        print(f"  {dataset_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} datasets compliant ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All datasets follow the standard format!")
        print("\nStandard Format Verified:")
        print("  ✅ id: str - Unique identifier")
        print("  ✅ text: str|null - Text content")
        print("  ✅ image_path: str|null - Path to image file")
        print("  ✅ audio_path: str|null - Path to audio file")
        print("\nBenefits:")
        print("  🔧 Consistent data loading across all examples")
        print("  🎯 Unified prompt building interface")
        print("  📊 Easy modality detection and routing")
        print("  🧪 Simplified testing and validation")
        return True
    else:
        print(f"\n⚠️  {total - passed} datasets need standardization")
        return False

def check_legacy_fields():
    """Check for legacy field names that need updating."""
    
    print("\n🔍 Checking for Legacy Field Names")
    print("=" * 50)
    
    root_dir = Path(__file__).parent.parent.parent
    examples_dir = root_dir / "some" / "examples"
    legacy_patterns = {
        "prompt_text": "text",
        "additional_text": "text",
        "audio_url": "audio_path",
        "image_url": "image_path"
    }

    # Check Python files for legacy field usage
    python_files = [
        examples_dir / "vision_extraction" / "product_prompt.py",
        examples_dir / "audio_extraction" / "audio_prompt.py",
        examples_dir / "multimodal_extraction" / "multimodal_prompt.py"
    ]
    
    legacy_found = False
    for py_file in python_files:
        if py_file.exists():
            content = py_file.read_text()
            for legacy_field, new_field in legacy_patterns.items():
                if f'"{legacy_field}"' in content or f"'{legacy_field}'" in content:
                    print(f"⚠️  Found legacy field '{legacy_field}' in {py_file.name}")
                    print(f"   Should be updated to '{new_field}'")
                    legacy_found = True
    
    if not legacy_found:
        print("✅ No legacy field names found in Python files")
    
    return not legacy_found

if __name__ == "__main__":
    print("Testing dataset standardization...")
    
    # Test dataset format compliance
    format_success = main()
    
    # Check for legacy field usage
    legacy_success = check_legacy_fields()
    
    overall_success = format_success and legacy_success
    
    if overall_success:
        print("\n🎉 Dataset standardization complete!")
    else:
        print("\n⚠️  Some standardization issues found")
    
    sys.exit(0 if overall_success else 1)
