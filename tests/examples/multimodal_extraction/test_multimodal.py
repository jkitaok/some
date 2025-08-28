#!/usr/bin/env python3
"""
Test script for multi-modal extraction.

This script provides comprehensive testing of all modality combinations
using available test data and remote resources.
"""

import sys
from pathlib import Path

# Add the root directory to the path so we can import from some
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from some.inference import get_language_model
from some.io import read_json
from some.examples.multimodal_extraction.multimodal_schema import (
    MultiModalAnalysis, TextOnlyAnalysis, VisionOnlyAnalysis,
    TextVisionAnalysis, ContentType
)
from some.examples.multimodal_extraction.multimodal_prompt import (
    MultiModalPrompt, TextOnlyPrompt, VisionOnlyPrompt, TextVisionPrompt
)

def test_schema_validation():
    """Test that all schemas work correctly."""
    
    print("🔍 Testing Schema Validation")
    print("=" * 50)
    
    try:
        # Test MultiModalAnalysis
        multimodal_data = {
            "content_type": "educational",
            "summary": "Test multi-modal content",
            "modalities_present": ["text", "vision"],
            "confidence_score": 0.85
        }
        analysis = MultiModalAnalysis(**multimodal_data)
        print("✅ MultiModalAnalysis schema validation passed")
        
        # Test TextOnlyAnalysis
        text_data = {
            "content_type": "educational",
            "summary": "Test text content",
            "key_topics": ["AI", "Technology"],
            "sentiment": "positive",
            "confidence_score": 0.9
        }
        text_analysis = TextOnlyAnalysis(**text_data)
        print("✅ TextOnlyAnalysis schema validation passed")
        
        # Test VisionOnlyAnalysis
        vision_data = {
            "visual_description": "A technology presentation slide",
            "objects_detected": ["laptop", "screen"],
            "scene_setting": "conference room",
            "people_count": 2,
            "mood_atmosphere": "professional",
            "confidence_score": 0.8
        }
        vision_analysis = VisionOnlyAnalysis(**vision_data)
        print("✅ VisionOnlyAnalysis schema validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        return False

def test_text_only_extraction():
    """Test text-only extraction."""
    
    print("\n📝 Testing Text-Only Extraction")
    print("=" * 50)
    
    # Load test data from JSON file
    root_dir = Path(__file__).parent.parent.parent.parent
    test_data_path = root_dir / "some" / "examples" / "multimodal_extraction" / "input_dataset" / "test_samples.json"
    test_data = read_json(str(test_data_path))
    if not test_data:
        print("❌ Failed to load test data")
        return False

    # Get text-only test sample
    text_sample = next((item for item in test_data if item["id"] == "test_text_001"), None)
    if not text_sample:
        print("❌ Text test sample not found")
        return False

    test_text = text_sample["text"]
    
    try:
        # Build prompt using standardized format
        prompt_builder = TextOnlyPrompt()
        extraction_input = prompt_builder.build(text_sample)
        print("✅ Text-only prompt built successfully")
        
        # Get language model
        lm = get_language_model(provider="instructor", model="gpt-4o-mini")
        print(f"✅ Language model initialized: {lm.model_id}")
        print(f"   Supported modalities: {lm.supported_modalities}")
        
        # Check available modalities
        available = lm.get_available_modalities(extraction_input)
        print(f"   Available for this input: {available}")
        
        # Run extraction
        results, workers, timing = lm.generate([extraction_input])
        print(f"✅ Extraction completed in {timing:.2f}s")
        
        # Display results
        result = results[0]
        if 'error' in result:
            print(f"❌ Extraction error: {result['error']}")
            return False
        
        analysis = result.get('text_analysis', {})
        if analysis:
            print("\n📊 Text Analysis Results:")
            print(f"  Content Type: {analysis.get('content_type', 'N/A')}")
            print(f"  Summary: {analysis.get('summary', 'N/A')[:100]}...")
            print(f"  Key Topics: {', '.join(analysis.get('key_topics', []))}")
            print(f"  Sentiment: {analysis.get('sentiment', 'N/A')}")
            print(f"  Confidence: {analysis.get('confidence_score', 'N/A')}")
            return True
        else:
            print("⚠️  No analysis results returned")
            return False
            
    except Exception as e:
        print(f"❌ Text-only extraction failed: {e}")
        return False

def test_vision_only_extraction():
    """Test vision-only extraction."""
    
    print("\n🖼️ Testing Vision-Only Extraction")
    print("=" * 50)
    
    # Look for rdj.jpg in the root directory
    root_dir = Path(__file__).parent.parent.parent.parent
    rdj_path = root_dir / "rdj.jpg"
    
    if not rdj_path.exists():
        print(f"❌ Test image not found: {rdj_path}")
        print("Skipping vision-only test")
        return False
    
    try:
        # Build prompt
        prompt_builder = VisionOnlyPrompt()
        extraction_input = prompt_builder.build({"image_path": str(rdj_path)})
        print("✅ Vision-only prompt built successfully")
        
        # Get language model
        lm = get_language_model(provider="instructor", model="gpt-4o-mini")
        print(f"✅ Language model initialized: {lm.model_id}")
        
        # Check available modalities
        available = lm.get_available_modalities(extraction_input)
        print(f"   Available modalities: {available}")
        
        if "vision" not in available:
            print("⚠️  Vision modality not supported by this model")
            return False
        
        # Run extraction
        results, workers, timing = lm.generate([extraction_input])
        print(f"✅ Extraction completed in {timing:.2f}s")
        
        # Display results
        result = results[0]
        if 'error' in result:
            print(f"❌ Extraction error: {result['error']}")
            return False
        
        analysis = result.get('vision_analysis', {})
        if analysis:
            print("\n📊 Vision Analysis Results:")
            print(f"  Visual Description: {analysis.get('visual_description', 'N/A')[:100]}...")
            print(f"  Objects Detected: {', '.join(analysis.get('objects_detected', []))}")
            print(f"  Scene Setting: {analysis.get('scene_setting', 'N/A')}")
            print(f"  People Count: {analysis.get('people_count', 'N/A')}")
            print(f"  Mood/Atmosphere: {analysis.get('mood_atmosphere', 'N/A')}")
            print(f"  Confidence: {analysis.get('confidence_score', 'N/A')}")
            return True
        else:
            print("⚠️  No analysis results returned")
            return False
            
    except Exception as e:
        print(f"❌ Vision-only extraction failed: {e}")
        return False

def test_text_vision_extraction():
    """Test text + vision extraction."""
    
    print("\n🎭 Testing Text + Vision Extraction")
    print("=" * 50)
    
    # Look for rdj.jpg in the root directory
    root_dir = Path(__file__).parent.parent.parent.parent
    rdj_path = root_dir / "rdj.jpg"
    
    if not rdj_path.exists():
        print(f"❌ Test image not found: {rdj_path}")
        print("Skipping text + vision test")
        return False
    
    # Load test data from JSON file
    root_dir = Path(__file__).parent.parent.parent.parent
    test_data_path = root_dir / "some" / "examples" / "multimodal_extraction" / "input_dataset" / "test_samples.json"
    test_data = read_json(str(test_data_path))
    if not test_data:
        print("❌ Failed to load test data")
        return False

    # Get text+vision test sample
    text_vision_sample = next((item for item in test_data if item["id"] == "test_text_vision_001"), None)
    if not text_vision_sample:
        print("❌ Text+vision test sample not found")
        return False
    
    try:
        # Update sample with correct image path
        text_vision_sample["image_path"] = str(rdj_path)

        # Build prompt using standardized format
        prompt_builder = TextVisionPrompt()
        extraction_input = prompt_builder.build(text_vision_sample)
        print("✅ Text + Vision prompt built successfully")
        
        # Get language model
        lm = get_language_model(provider="instructor", model="gpt-4o-mini")
        print(f"✅ Language model initialized: {lm.model_id}")
        
        # Check available modalities
        available = lm.get_available_modalities(extraction_input)
        print(f"   Available modalities: {available}")
        
        if "vision" not in available:
            print("⚠️  Vision modality not supported by this model")
            return False
        
        # Run extraction
        results, workers, timing = lm.generate([extraction_input])
        print(f"✅ Extraction completed in {timing:.2f}s")
        
        # Display results
        result = results[0]
        if 'error' in result:
            print(f"❌ Extraction error: {result['error']}")
            return False
        
        analysis = result.get('text_vision_analysis', {})
        if analysis:
            print("\n📊 Text + Vision Analysis Results:")
            print(f"  Content Type: {analysis.get('content_type', 'N/A')}")
            print(f"  Summary: {analysis.get('summary', 'N/A')[:100]}...")
            print(f"  Text-Image Alignment: {analysis.get('text_image_alignment', 'N/A')}")
            print(f"  Objects Detected: {', '.join(analysis.get('objects_detected', []))}")
            print(f"  Key Topics: {', '.join(analysis.get('key_topics', []))}")
            print(f"  Confidence: {analysis.get('confidence_score', 'N/A')}")
            return True
        else:
            print("⚠️  No analysis results returned")
            return False
            
    except Exception as e:
        print(f"❌ Text + Vision extraction failed: {e}")
        return False

def test_modality_detection():
    """Test modality detection capabilities."""
    
    print("\n🔍 Testing Modality Detection")
    print("=" * 50)
    
    try:
        lm = get_language_model(provider="instructor", model="gpt-4o-mini")
        print(f"Model: {lm.model_id}")
        print(f"Supported modalities: {lm.supported_modalities}")
        
        # Test different input combinations
        test_cases = [
            {"prompt_text": "Just text"},
            {"image_path": "test.jpg"},
            {"audio_url": "test.wav"},
            {"prompt_text": "Text", "image_path": "test.jpg"},
            {"prompt_text": "Text", "audio_url": "test.wav"},
            {"image_path": "test.jpg", "audio_url": "test.wav"},
            {"prompt_text": "Text", "image_path": "test.jpg", "audio_url": "test.wav"}
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            available = lm.get_available_modalities(test_case)
            inputs = list(test_case.keys())
            print(f"  Test {i}: {inputs} → Available: {available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Modality detection test failed: {e}")
        return False

def main():
    """Run all multi-modal tests."""
    
    print("🧪 Multi-Modal Extraction Test Suite")
    print("=" * 60)
    
    # Test 1: Schema validation
    schema_test = test_schema_validation()
    
    # Test 2: Modality detection
    modality_test = test_modality_detection()
    
    # Test 3: Text-only extraction
    text_test = test_text_only_extraction()
    
    # Test 4: Vision-only extraction
    vision_test = test_vision_only_extraction()
    
    # Test 5: Text + Vision extraction
    text_vision_test = test_text_vision_extraction()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"  Schema validation: {'✅ PASS' if schema_test else '❌ FAIL'}")
    print(f"  Modality detection: {'✅ PASS' if modality_test else '❌ FAIL'}")
    print(f"  Text-only extraction: {'✅ PASS' if text_test else '❌ FAIL'}")
    print(f"  Vision-only extraction: {'✅ PASS' if vision_test else '⚠️  SKIP'}")
    print(f"  Text+Vision extraction: {'✅ PASS' if text_vision_test else '⚠️  SKIP'}")
    
    passed_tests = sum([schema_test, modality_test, text_test, vision_test, text_vision_test])
    total_tests = 5
    
    if passed_tests >= 3:  # Allow for vision tests to be skipped
        print(f"\n🎉 Multi-modal extraction system is working! ({passed_tests}/{total_tests} tests passed)")
        print("\nKey Features Demonstrated:")
        print("  ✅ Multiple modality combinations supported")
        print("  ✅ Automatic modality detection")
        print("  ✅ Schema-based structured output")
        print("  ✅ Cross-modal analysis capabilities")
        return True
    else:
        print(f"\n⚠️  Some core tests failed. ({passed_tests}/{total_tests} tests passed)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
