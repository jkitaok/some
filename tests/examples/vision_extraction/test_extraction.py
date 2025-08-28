#!/usr/bin/env python3
"""
Test script for vision product extraction.

This script provides a simple way to test the vision extraction functionality
with minimal setup. It can work with the existing rdj.jpg file or any other image.
"""

import os
import sys
from pathlib import Path

# Add the root directory to the path so we can import from some
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from some.inference import get_language_model
from some.examples.vision_extraction.product_schema import ProductDetails
from some.examples.vision_extraction.product_prompt import ProductExtractionPrompt

def test_with_rdj_image():
    """Test extraction using the rdj.jpg image from the root directory."""
    
    # Look for rdj.jpg in the root directory
    root_dir = Path(__file__).parent.parent.parent.parent
    rdj_path = root_dir / "rdj.jpg"
    
    if not rdj_path.exists():
        print(f"❌ Test image not found: {rdj_path}")
        print("Please ensure rdj.jpg exists in the root directory, or provide your own image.")
        return False
    
    print(f"🖼️  Testing with image: {rdj_path}")
    
    # Create test item
    test_item = {
        "image_path": str(rdj_path),
        "additional_text": "This is a test image of a person for vision extraction testing."
    }
    
    # Build extraction input
    try:
        prompt_builder = ProductExtractionPrompt()
        extraction_input = prompt_builder.build(test_item)
        print("✅ Successfully built extraction prompt")
    except Exception as e:
        print(f"❌ Failed to build prompt: {e}")
        return False
    
    # Test language model initialization
    try:
        print("🤖 Attempting to initialize language model...")
        
        # Try instructor first
        try:
            lm = get_language_model(provider="instructor", model="openai/gpt-4o-mini")
            provider_name = "instructor (gpt-4o-mini)"
            print(f"✅ Successfully initialized {provider_name}")
        except Exception as e:
            print(f"⚠️  Instructor provider failed: {e}")
            print("🔄 Trying OpenAI provider...")
            
            lm = get_language_model(provider="openai", model="gpt-4o-mini")
            provider_name = "openai (gpt-4o-mini)"
            print(f"✅ Successfully initialized {provider_name}")
            
    except Exception as e:
        print(f"❌ Failed to initialize language model: {e}")
        print("Please check your API key configuration:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return False
    
    # Run extraction
    try:
        print("🔍 Running extraction...")
        results, workers, timing = lm.generate([extraction_input])
        print(f"✅ Extraction completed in {timing:.2f}s using {workers} workers")
        
        # Display results
        result = results[0]
        if 'error' in result:
            print(f"❌ Extraction error: {result['error']}")
            return False
        
        product_details = result.get('product_details', {})
        if not product_details:
            print("⚠️  No product details extracted")
            return False
        
        print("\n📦 Extracted Product Details:")
        print(f"  Name: {product_details.get('name', 'N/A')}")
        print(f"  Category: {product_details.get('category', 'N/A')}")
        print(f"  Description: {product_details.get('description', 'N/A')}")
        print(f"  Confidence: {product_details.get('confidence_score', 'N/A')}")
        print(f"  Image Quality: {product_details.get('image_quality', 'N/A')}")
        
        if product_details.get('key_features'):
            print(f"  Features: {', '.join(product_details['key_features'])}")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def test_schema_validation():
    """Test that the ProductDetails schema works correctly."""
    
    print("🔍 Testing schema validation...")
    
    try:
        # Test valid data
        valid_data = {
            "name": "Test Product",
            "category": "electronics",
            "confidence_score": 0.85,
            "image_quality": "clear",
            "key_features": ["feature1", "feature2"]
        }
        
        product = ProductDetails(**valid_data)
        print("✅ Schema validation passed for valid data")
        
        # Test invalid data
        try:
            invalid_data = {
                "name": "Test Product",
                "category": "invalid_category",  # Should fail
                "confidence_score": 1.5,  # Should fail (>1.0)
                "image_quality": "clear"
            }
            ProductDetails(**invalid_data)
            print("❌ Schema validation should have failed for invalid data")
            return False
        except Exception:
            print("✅ Schema validation correctly rejected invalid data")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema validation test failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🧪 Vision Product Extraction Test Suite")
    print("=" * 50)
    
    # Test 1: Schema validation
    schema_test = test_schema_validation()
    
    print("\n" + "-" * 50)
    
    # Test 2: Image extraction
    extraction_test = test_with_rdj_image()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Schema validation: {'✅ PASS' if schema_test else '❌ FAIL'}")
    print(f"  Image extraction: {'✅ PASS' if extraction_test else '❌ FAIL'}")
    
    if schema_test and extraction_test:
        print("\n🎉 All tests passed! The vision extraction system is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
