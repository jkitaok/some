"""
Audio Extraction Example

This demonstrates how to use the some library for extracting structured information
from audio files using OpenAI's audio-capable models and the instructor library.

The example includes:
1. Audio content analysis and transcription
2. Speaker identification and analysis
3. Key information extraction (topics, names, decisions)
4. Quality evaluation of extraction results
5. Support for different audio types (meetings, podcasts, interviews, etc.)

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
from .audio_schema import AudioAnalysis, AudioEvaluation, AudioSummary
from .audio_prompt import AudioAnalysisPrompt, AudioEvaluationPrompt, AudioSummaryPrompt

def load_sample_data(data_path: str = "input_dataset/sample_audio.json") -> List[Dict[str, Any]]:
    """Load sample audio data from JSON file."""
    current_dir = Path(__file__).parent
    full_path = current_dir / data_path

    # Use io.py to read JSON data
    data = read_json(str(full_path))
    if data is None:
        raise FileNotFoundError(f"Sample data file not found or invalid: {full_path}")

    # Convert relative paths to absolute paths
    for item in data:
        if 'audio_path' in item:
            audio_path = current_dir / item['audio_path']
            item['audio_path'] = str(audio_path)

    return data

def format_evaluation_record(extraction_input: Dict[str, Any], extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Format a record for evaluation by combining input and output."""
    result = {
        "extraction_result": extraction_result.get("audio_analysis"),
        "expected_content": extraction_input.get("expected_content")
    }
    
    # Add audio source
    if "audio_url" in extraction_input:
        result["audio_url"] = extraction_input["audio_url"]
    elif "audio_path" in extraction_input:
        result["audio_path"] = extraction_input["audio_path"]
    
    return result

def print_audio_analysis(analysis: Dict[str, Any], audio_id: str = ""):
    """Pretty print audio analysis results."""
    if not analysis:
        print(f"  {audio_id}: No analysis extracted")
        return
    
    print(f"  {audio_id}:")
    print(f"    Type: {analysis.get('audio_type', 'N/A')}")
    print(f"    Language: {analysis.get('language', 'N/A')}")
    print(f"    Speakers: {analysis.get('speaker_count', 'N/A')}")
    print(f"    Quality: {analysis.get('audio_quality', 'N/A')}")
    print(f"    Confidence: {analysis.get('confidence_score', 'N/A')}")
    
    # Show summary (truncated)
    summary = analysis.get('summary', '')
    if summary:
        print(f"    Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}")
    
    # Show main topics
    topics = analysis.get('main_topics', [])
    if topics:
        print(f"    Topics: {', '.join(topics[:3])}{'...' if len(topics) > 3 else ''}")
    
    # Show transcript preview
    transcript = analysis.get('transcript', '')
    if transcript:
        print(f"    Transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")

def main():
    """Main function for running the audio extraction example."""
    
    print("🎵 Audio Extraction Example")
    print("=" * 60)
    
    # Load sample data
    try:
        sample_data = load_sample_data()
        print(f"Loaded {len(sample_data)} sample audio files")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure sample data is available in input_dataset/")
        return
    
    # Filter to available audio sources (URLs or existing files)
    available_items = []
    for item in sample_data:
        if item.get('audio_url'):
            # URL-based audio is always "available"
            available_items.append(item)
        elif item.get('audio_path') and os.path.exists(item['audio_path']):
            available_items.append(item)
        else:
            audio_source = item.get('audio_path', 'No path specified')
            print(f"Warning: Audio not found, skipping {item['id']}: {audio_source}")
    
    if not available_items:
        print("No audio sources found. The example includes a remote URL that should work.")
        print("Check your internet connection or add local audio files to input_dataset/audio_files/")
        return
    
    print(f"Processing {len(available_items)} items with available audio")
    
    # Build extraction inputs with media validation
    extraction_inputs = []
    for item in available_items:
        try:
            validated_item = validate_media_input(item)
            extraction_input = AudioAnalysisPrompt().build(validated_item)
            extraction_inputs.append(extraction_input)
        except Exception as e:
            print(f"Warning: Failed to process item {item.get('id', 'unknown')}: {e}")
            continue
    
    # Get language model - try instructor first, then OpenAI
    try:
        print("Attempting to use instructor provider for audio processing...")
        lm = get_language_model(provider="instructor", model="openai/gpt-4o-audio-preview")
        provider_name = "instructor (gpt-4o-audio-preview)"
    except Exception as e:
        print(f"Instructor provider failed: {e}")
        print("Falling back to OpenAI provider...")
        try:
            lm = get_language_model(provider="openai", model="gpt-4o-audio-preview")
            provider_name = "openai (gpt-4o-audio-preview)"
        except Exception as e:
            print(f"OpenAI provider also failed: {e}")
            print("Please ensure you have:")
            print("1. OpenAI API key configured: export OPENAI_API_KEY='your-key'")
            print("2. Access to audio models (gpt-4o-audio-preview)")
            print("3. Instructor library installed: pip install instructor")
            return
    
    print(f"Using {provider_name} language model")
    
    # Setup metrics collection
    llm_collector = LLMMetricsCollector(
        name="audio_extraction",
        cost_per_input_token=0.15/1000000,  # Approximate audio processing cost
        cost_per_output_token=0.6/1000000
    )
    
    # Run audio extraction
    print("\n🔍 Extracting audio content...")
    extraction_results, effective_workers, extraction_time = lm.generate(extraction_inputs)
    llm_collector.add_inference_time(extraction_time)
    
    print(f"Extraction completed using {effective_workers} workers in {extraction_time:.4f}s")
    
    # Display extraction results
    print("\n🎵 Audio Extraction Results:")
    for i, (item, result) in enumerate(zip(available_items, extraction_results)):
        audio_id = item.get('id', f'audio_{i}')
        if 'error' in result:
            print(f"  {audio_id}: Error - {result['error']}")
        else:
            print_audio_analysis(result.get('audio_analysis'), audio_id)
    
    # Run evaluation if we have successful extractions
    successful_extractions = [(inp, res) for inp, res in zip(available_items, extraction_results) 
                            if 'error' not in res and res.get('audio_analysis')]
    
    if successful_extractions:
        print(f"\n🔬 Evaluating {len(successful_extractions)} successful extractions...")
        
        # Format evaluation inputs
        evaluation_inputs = [
            AudioEvaluationPrompt().build(format_evaluation_record(inp, res))
            for inp, res in successful_extractions
        ]
        
        # Run evaluation
        evaluation_results, _, evaluation_time = lm.generate(evaluation_inputs)
        llm_collector.add_inference_time(evaluation_time)
        
        # Display evaluation results
        print("\n📊 Evaluation Results:")
        for i, (item, eval_result) in enumerate(zip([inp for inp, _ in successful_extractions], evaluation_results)):
            audio_id = item.get('id', f'audio_{i}')
            if 'error' in eval_result:
                print(f"  {audio_id}: Evaluation error - {eval_result['error']}")
            else:
                evaluation = eval_result.get('evaluation', {})
                print(f"  {audio_id}:")
                print(f"    Overall Quality: {evaluation.get('overall_quality', 'N/A')}")
                print(f"    Transcript Accuracy: {evaluation.get('transcript_accuracy', 'N/A')}")
                print(f"    Speaker ID: {evaluation.get('speaker_identification', 'N/A')}")
                print(f"    Content Understanding: {evaluation.get('content_understanding', 'N/A')}")
                if evaluation.get('reasoning'):
                    print(f"    Reasoning: {evaluation['reasoning'][:100]}...")
    else:
        evaluation_results = []
        print("\n⚠️  No successful extractions to evaluate")
    
    # Collect and display metrics
    all_results = extraction_results + evaluation_results
    llm_metrics = llm_collector.collect_metrics(all_results)
    
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE METRICS")
    print("=" * 60)
    print(llm_collector.format_summary(llm_metrics))
    
    # Schema-based metrics for evaluations
    if evaluation_results:
        evaluation_data = [r.get('evaluation') for r in evaluation_results if r.get('evaluation')]
        if evaluation_data:
            schema_collector = SchemaMetricsCollector(AudioEvaluation, "evaluation_quality")
            schema_metrics = schema_collector.collect_metrics(evaluation_data)
            
            print("\n" + "=" * 60)
            print("🎯 EXTRACTION QUALITY METRICS")
            print("=" * 60)
            print(schema_collector.format_summary(schema_metrics))
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"Total audio files processed: {len(available_items)}")
    print(f"Successful extractions: {len(successful_extractions)}")
    print(f"Success rate: {len(successful_extractions)/len(available_items)*100:.1f}%")
    print(f"Total LLM calls: {llm_metrics['total_items']}")
    print(f"Total cost: ${llm_metrics['total_cost']:.6f}")
    print(f"Total time: {llm_metrics['total_inference_time']:.4f}s")
    
    print("\n✅ Audio extraction analysis completed!")
    
    return {
        "extraction_results": extraction_results,
        "evaluation_results": evaluation_results,
        "llm_metrics": llm_metrics
    }

if __name__ == "__main__":
    main()
