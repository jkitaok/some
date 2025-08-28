#!/usr/bin/env python3
"""
Test script for audio extraction.

This script provides a simple way to test the audio extraction functionality
using the remote Gettysburg Address audio file from the instructor library.
"""

import os
import sys
from pathlib import Path

# Add the root directory to the path so we can import from some
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from some.inference import get_language_model
from some.examples.audio_extraction.audio_schema import AudioAnalysis, AudioSummary
from some.examples.audio_extraction.audio_prompt import AudioAnalysisPrompt, AudioSummaryPrompt

def test_with_gettysburg_audio():
    """Test extraction using the Gettysburg Address audio from instructor library."""
    
    # Use the remote audio URL from instructor library
    audio_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"
    
    print(f"🎵 Testing with remote audio: {audio_url}")
    
    # Create test item
    test_item = {
        "audio_url": audio_url,
        "context": "Historical speech - The Gettysburg Address by Abraham Lincoln",
        "expected_type": "speech"
    }
    
    # Build extraction input
    try:
        prompt_builder = AudioAnalysisPrompt()
        extraction_input = prompt_builder.build(test_item)
        print("✅ Successfully built audio analysis prompt")
    except Exception as e:
        print(f"❌ Failed to build prompt: {e}")
        return False
    
    # Test language model initialization
    try:
        print("🤖 Attempting to initialize audio language model...")
        
        # Try instructor with audio-capable model first
        try:
            lm = get_language_model(provider="instructor", model="gpt-4o-audio-preview")
            provider_name = "instructor (gpt-4o-audio-preview)"
            print(f"✅ Successfully initialized {provider_name}")
        except Exception as e:
            print(f"⚠️  Instructor provider failed: {e}")
            print("🔄 Trying OpenAI provider...")

            try:
                lm = get_language_model(provider="openai", model="gpt-4o-audio-preview")
                provider_name = "openai (gpt-4o-audio-preview)"
                print(f"✅ Successfully initialized {provider_name}")
            except Exception as e:
                print(f"⚠️  OpenAI provider also failed: {e}")
                print("🔄 Trying with regular models...")

                lm = get_language_model(provider="instructor", model="gpt-4o-mini")
                provider_name = "instructor (gpt-4o-mini - limited audio support)"
                print(f"✅ Successfully initialized {provider_name}")
            
    except Exception as e:
        print(f"❌ Failed to initialize language model: {e}")
        print("Please check:")
        print("  1. OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        print("  2. Access to audio models (gpt-4o-audio-preview)")
        print("  3. Instructor library: pip install instructor")
        return False
    
    # Run extraction
    try:
        print("🔍 Running audio extraction...")
        results, workers, timing = lm.generate([extraction_input])
        print(f"✅ Extraction completed in {timing:.2f}s using {workers} workers")
        
        # Display results
        result = results[0]
        if 'error' in result:
            print(f"❌ Extraction error: {result['error']}")
            return False
        
        audio_analysis = result.get('audio_analysis', {})
        if not audio_analysis:
            print("⚠️  No audio analysis extracted")
            return False
        
        print("\n🎵 Extracted Audio Analysis:")
        print(f"  Type: {audio_analysis.get('audio_type', 'N/A')}")
        print(f"  Language: {audio_analysis.get('language', 'N/A')}")
        print(f"  Speakers: {audio_analysis.get('speaker_count', 'N/A')}")
        print(f"  Quality: {audio_analysis.get('audio_quality', 'N/A')}")
        print(f"  Confidence: {audio_analysis.get('confidence_score', 'N/A')}")
        
        # Show summary
        summary = audio_analysis.get('summary', '')
        if summary:
            print(f"  Summary: {summary[:150]}{'...' if len(summary) > 150 else ''}")
        
        # Show transcript preview
        transcript = audio_analysis.get('transcript', '')
        if transcript:
            print(f"  Transcript: {transcript[:200]}{'...' if len(transcript) > 200 else ''}")
        
        # Show topics
        topics = audio_analysis.get('main_topics', [])
        if topics:
            print(f"  Topics: {', '.join(topics)}")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def test_schema_validation():
    """Test that the AudioAnalysis schema works correctly."""
    
    print("🔍 Testing schema validation...")
    
    try:
        # Test valid data
        valid_data = {
            "audio_type": "speech",
            "language": "English",
            "summary": "Test audio summary",
            "transcript": "Test transcript content",
            "speaker_count": 1,
            "audio_quality": "good",
            "confidence_score": 0.85,
            "overall_tone": "formal",
            "sentiment": "neutral"
        }
        
        analysis = AudioAnalysis(**valid_data)
        print("✅ Schema validation passed for valid data")
        
        # Test invalid data
        try:
            invalid_data = {
                "audio_type": "invalid_type",  # Should fail
                "language": "English",
                "summary": "Test summary",
                "transcript": "Test transcript",
                "speaker_count": 1,
                "audio_quality": "good",
                "confidence_score": 1.5,  # Should fail (>1.0)
                "overall_tone": "formal",
                "sentiment": "neutral"
            }
            AudioAnalysis(**invalid_data)
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
    
    print("🧪 Audio Extraction Test Suite")
    print("=" * 50)
    
    # Test 1: Schema validation
    schema_test = test_schema_validation()
    
    print("\n" + "-" * 50)
    
    # Test 2: Audio extraction
    extraction_test = test_with_gettysburg_audio()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Schema validation: {'✅ PASS' if schema_test else '❌ FAIL'}")
    print(f"  Audio extraction: {'✅ PASS' if extraction_test else '❌ FAIL'}")
    
    if schema_test and extraction_test:
        print("\n🎉 All tests passed! The audio extraction system is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
