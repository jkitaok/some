"""
Multi-Modal Extraction Example

This demonstrates how to use the some library for extracting structured information
from content that combines text, vision, and audio modalities. The example includes:

1. Analysis of individual modalities (text-only, vision-only, audio-only)
2. Analysis of dual modality combinations (text+vision, text+audio, vision+audio)
3. Comprehensive multi-modal analysis (text+vision+audio)
4. Cross-modal insights and alignment detection
5. Performance metrics and quality assessment

Run this as a Python module or import the functions for use in your own code.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List

from some.inference import get_language_model
from some.metrics import LLMMetricsCollector, SchemaMetricsCollector
from some.io import read_json
from some.media import validate_media_input
from .multimodal_schema import (
    MultiModalAnalysis, TextOnlyAnalysis, VisionOnlyAnalysis, AudioOnlyAnalysis,
    TextVisionAnalysis, TextAudioAnalysis, VisionAudioAnalysis
)
from .multimodal_prompt import (
    MultiModalPrompt, TextOnlyPrompt, VisionOnlyPrompt, AudioOnlyPrompt,
    TextVisionPrompt, TextAudioPrompt, VisionAudioPrompt
)

def load_sample_data(data_path: str = "input_dataset/sample_multimodal.json") -> List[Dict[str, Any]]:
    """Load sample multi-modal data from JSON file."""
    current_dir = Path(__file__).parent
    full_path = current_dir / data_path

    # Use io.py to read JSON data
    data = read_json(str(full_path))
    if data is None:
        raise FileNotFoundError(f"Sample data file not found or invalid: {full_path}")

    # Convert relative paths to absolute paths
    for item in data:
        if 'image_path' in item:
            image_path = current_dir / item['image_path']
            item['image_path'] = str(image_path)
        if 'audio_path' in item:
            audio_path = current_dir / item['audio_path']
            item['audio_path'] = str(audio_path)

    return data

def determine_prompt_builder(modalities: List[str]):
    """Determine which prompt builder to use based on modalities."""
    modality_set = set(modalities)
    
    if modality_set == {"text"}:
        return TextOnlyPrompt()
    elif modality_set == {"vision"}:
        return VisionOnlyPrompt()
    elif modality_set == {"audio"}:
        return AudioOnlyPrompt()
    elif modality_set == {"text", "vision"}:
        return TextVisionPrompt()
    elif modality_set == {"text", "audio"}:
        return TextAudioPrompt()
    elif modality_set == {"vision", "audio"}:
        return VisionAudioPrompt()
    elif modality_set == {"text", "vision", "audio"}:
        return MultiModalPrompt()
    else:
        # Default to multi-modal for any other combination
        return MultiModalPrompt()

def print_analysis_summary(analysis: Dict[str, Any], sample_id: str, modalities: List[str]):
    """Pretty print analysis results."""
    if not analysis:
        print(f"  {sample_id}: No analysis extracted")
        return
    
    print(f"  {sample_id} ({'+'.join(modalities)}):")
    
    # Common fields across all schemas
    content_type = analysis.get('content_type', 'N/A')
    summary = analysis.get('summary', '')
    confidence = analysis.get('confidence_score', 'N/A')
    
    print(f"    Content Type: {content_type}")
    print(f"    Confidence: {confidence}")
    
    if summary:
        print(f"    Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}")
    
    # Modality-specific fields
    if 'modalities_present' in analysis:
        print(f"    Modalities Detected: {analysis['modalities_present']}")
    
    if 'key_topics' in analysis:
        topics = analysis['key_topics']
        if topics:
            print(f"    Key Topics: {', '.join(topics[:3])}{'...' if len(topics) > 3 else ''}")
    
    if 'objects_detected' in analysis:
        objects = analysis['objects_detected']
        if objects:
            print(f"    Objects: {', '.join(objects[:3])}{'...' if len(objects) > 3 else ''}")
    
    if 'speaker_count' in analysis:
        print(f"    Speakers: {analysis['speaker_count']}")
    
    # Cross-modal insights
    if 'modality_alignment' in analysis:
        print(f"    Modality Alignment: {analysis['modality_alignment']}")
    
    if 'unique_insights' in analysis:
        insights = analysis['unique_insights']
        if insights:
            print(f"    Unique Insights: {len(insights)} identified")

def main():
    """Main function for running the multi-modal extraction example."""
    
    print("🎭 Multi-Modal Extraction Example")
    print("=" * 60)
    
    # Load sample data
    try:
        sample_data = load_sample_data()
        print(f"Loaded {len(sample_data)} sample multi-modal items")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure sample data is available in input_dataset/")
        return
    
    # Filter to available items (check for existing files and URLs)
    available_items = []
    for item in sample_data:
        available = True
        
        # Check image availability
        if item.get('image_path') and not os.path.exists(item['image_path']):
            if not item.get('image_url'):
                print(f"Warning: Image not found for {item['id']}: {item['image_path']}")
                available = False
        
        # Check audio availability (URLs are always "available")
        if item.get('audio_path') and not os.path.exists(item['audio_path']):
            if not item.get('audio_url'):
                print(f"Warning: Audio not found for {item['id']}: {item['audio_path']}")
                available = False
        
        if available:
            available_items.append(item)
    
    if not available_items:
        print("No complete multi-modal samples found.")
        print("Note: Some samples use remote URLs which should work with internet connection.")
        return
    
    print(f"Processing {len(available_items)} items with available content")
    
    # Group items by modality combination for organized processing
    modality_groups = {}
    for item in available_items:
        modalities = tuple(sorted(item.get('modalities', [])))
        if modalities not in modality_groups:
            modality_groups[modalities] = []
        modality_groups[modalities].append(item)
    
    print(f"Found {len(modality_groups)} different modality combinations:")
    for modalities, items in modality_groups.items():
        print(f"  {'+'.join(modalities)}: {len(items)} items")
    
    # Get language model
    try:
        print("\nAttempting to initialize multi-modal language model...")
        # Try instructor with vision/audio capable model first
        try:
            lm = get_language_model(provider="instructor", model="gpt-4o")
            provider_name = "instructor (gpt-4o)"
        except Exception as e:
            print(f"GPT-4o failed: {e}")
            try:
                lm = get_language_model(provider="instructor", model="gpt-4o-mini")
                provider_name = "instructor (gpt-4o-mini)"
            except Exception as e:
                print(f"Instructor failed: {e}")
                lm = get_language_model(provider="openai", model="gpt-4o-mini")
                provider_name = "openai (gpt-4o-mini)"
        
        print(f"✅ Using {provider_name}")
        print(f"   Supported modalities: {lm.supported_modalities}")
        
    except Exception as e:
        print(f"❌ Failed to initialize language model: {e}")
        print("Please ensure you have:")
        print("1. OpenAI API key configured")
        print("2. Access to multi-modal models")
        print("3. Instructor library installed")
        return
    
    # Setup metrics collection
    llm_collector = LLMMetricsCollector(
        name="multimodal_extraction",
        cost_per_input_token=0.15/1000000,
        cost_per_output_token=0.6/1000000
    )
    
    # Process each modality group
    all_results = []
    
    for modalities, items in modality_groups.items():
        print(f"\n🔍 Processing {'+'.join(modalities)} items ({len(items)} samples)...")
        
        # Build extraction inputs for this modality group
        prompt_builder = determine_prompt_builder(list(modalities))
        extraction_inputs = []
        
        for item in items:
            try:
                validated_item = validate_media_input(item)
                extraction_input = prompt_builder.build(validated_item)
                extraction_inputs.append(extraction_input)
            except Exception as e:
                print(f"  Warning: Failed to build prompt for {item['id']}: {e}")
        
        if not extraction_inputs:
            print(f"  No valid inputs for {'+'.join(modalities)} group")
            continue
        
        # Run extraction for this group
        try:
            results, workers, timing = lm.generate(extraction_inputs)
            llm_collector.add_inference_time(timing)
            
            print(f"  ✅ Completed {len(results)} extractions in {timing:.2f}s using {workers} workers")
            
            # Store results with metadata
            for i, (item, result) in enumerate(zip(items, results)):
                result['_metadata'] = {
                    'sample_id': item['id'],
                    'modalities': list(modalities),
                    'expected_analysis': item.get('expected_analysis', {})
                }
                all_results.append(result)
            
        except Exception as e:
            print(f"  ❌ Extraction failed for {'+'.join(modalities)} group: {e}")
    
    # Display results by modality group
    print("\n" + "=" * 60)
    print("📊 EXTRACTION RESULTS")
    print("=" * 60)
    
    for modalities, items in modality_groups.items():
        print(f"\n{'+'.join(modalities).upper()} ANALYSIS:")
        
        # Find results for this modality group
        group_results = [r for r in all_results 
                        if r.get('_metadata', {}).get('modalities') == list(modalities)]
        
        for result in group_results:
            metadata = result.get('_metadata', {})
            sample_id = metadata.get('sample_id', 'unknown')
            
            if 'error' in result:
                print(f"  {sample_id}: Error - {result['error']}")
            else:
                # Get the analysis result (key varies by schema)
                analysis = None
                for key in result.keys():
                    if key.endswith('_analysis') and not key.startswith('_'):
                        analysis = result[key]
                        break
                
                if analysis:
                    print_analysis_summary(analysis, sample_id, list(modalities))
                else:
                    print(f"  {sample_id}: No analysis data found")
    
    # Collect and display metrics
    llm_metrics = llm_collector.collect_metrics(all_results)
    
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE METRICS")
    print("=" * 60)
    print(llm_collector.format_summary(llm_metrics))
    
    # Summary statistics
    successful_extractions = [r for r in all_results if 'error' not in r]
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"Total samples processed: {len(available_items)}")
    print(f"Successful extractions: {len(successful_extractions)}")
    print(f"Success rate: {len(successful_extractions)/len(available_items)*100:.1f}%")
    print(f"Modality combinations tested: {len(modality_groups)}")
    print(f"Total LLM calls: {llm_metrics['total_items']}")
    print(f"Total cost: ${llm_metrics['total_cost']:.6f}")
    print(f"Total time: {llm_metrics['total_inference_time']:.4f}s")
    
    print("\n✅ Multi-modal extraction analysis completed!")
    
    return {
        "extraction_results": all_results,
        "llm_metrics": llm_metrics,
        "modality_groups": modality_groups
    }

if __name__ == "__main__":
    main()
