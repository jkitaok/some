"""
Vision Product Extraction Example

This demonstrates how to use the some library for extracting structured product information
from images using vision-language models. The example includes:

1. Product detail extraction from product images
2. Quality evaluation of extraction results
3. Performance metrics and cost analysis
4. Support for multiple vision-language models

Note: Sample product images referenced in this example were generated using FLUX.1-dev
(https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev) by Black Forest Labs.

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
from .product_schema import ProductDetails, ProductEvaluation
from .product_prompt import ProductExtractionPrompt, ProductEvaluationPrompt

def load_sample_data(data_path: str = "input_dataset/sample_products.json") -> List[Dict[str, Any]]:
    """Load sample product data from JSON file."""
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

    return data

def format_evaluation_record(extraction_input: Dict[str, Any], extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Format a record for evaluation by combining input and output."""
    return {
        "image_path": extraction_input["image_path"],
        "extraction_result": extraction_result.get("product_details"),
        "expected_details": extraction_input.get("expected_details")
    }

def print_product_details(product_details: Dict[str, Any], product_id: str = ""):
    """Pretty print product details."""
    if not product_details:
        print(f"  {product_id}: No details extracted")
        return
    
    print(f"  {product_id}:")
    print(f"    Name: {product_details.get('name', 'N/A')}")
    print(f"    Brand: {product_details.get('brand', 'N/A')}")
    print(f"    Category: {product_details.get('category', 'N/A')}")
    print(f"    Price: {product_details.get('price', 'N/A')} {product_details.get('currency', '')}")
    print(f"    Color: {product_details.get('color', 'N/A')}")
    print(f"    Features: {', '.join(product_details.get('key_features', []))}")
    print(f"    Confidence: {product_details.get('confidence_score', 'N/A')}")
    print(f"    Image Quality: {product_details.get('image_quality', 'N/A')}")

def main():
    """Main function for running the multimodal extraction example."""
    
    print("🖼️  Vision Product Extraction Example")
    print("=" * 60)
    
    # Load sample data
    try:
        sample_data = load_sample_data()
        print(f"Loaded {len(sample_data)} sample products")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure sample data and images are available in input_dataset/")
        return
    
    # Filter to only items with existing images for this demo
    available_items = []
    for item in sample_data:
        if os.path.exists(item['image_path']):
            available_items.append(item)
        else:
            print(f"Warning: Image not found, skipping {item['id']}: {item['image_path']}")
    
    if not available_items:
        print("No images found. Please add sample images to input_dataset/images/")
        print("You can use the existing rdj.jpg or add your own product images.")
        
        # Fallback to rdj.jpg if available
        rdj_path = Path(__file__).parent.parent.parent.parent / "rdj.jpg"
        if rdj_path.exists():
            print(f"Using fallback image: {rdj_path}")
            available_items = [{
                "id": "demo_person",
                "image_path": str(rdj_path),
                "additional_text": "Demo image for testing multimodal extraction"
            }]
        else:
            return
    
    print(f"Processing {len(available_items)} items with available images")
    
    # Build extraction inputs with media validation
    extraction_inputs = []
    for item in available_items:
        try:
            validated_item = validate_media_input(item)
            extraction_input = ProductExtractionPrompt().build(validated_item)
            extraction_inputs.append(extraction_input)
        except Exception as e:
            print(f"Warning: Failed to process item {item.get('id', 'unknown')}: {e}")
            continue
    
    # Get language model - try instructor first, fallback to openai
    try:
        print("Attempting to use instructor provider...")
        lm = get_language_model(provider="instructor", model="openai/gpt-4o-mini")
        provider_name = "instructor (gpt-4o-mini)"
    except Exception as e:
        print(f"Instructor provider failed: {e}")
        print("Falling back to OpenAI provider...")
        try:
            lm = get_language_model(provider="openai", model="gpt-4o-mini")
            provider_name = "openai (gpt-4o-mini)"
        except Exception as e:
            print(f"OpenAI provider also failed: {e}")
            print("Please ensure you have proper API keys configured.")
            return
    
    print(f"Using {provider_name} language model")
    
    # Setup metrics collection
    llm_collector = LLMMetricsCollector(
        name="vision_product_extraction",
        cost_per_input_token=0.15/1000000,  # GPT-4o-mini pricing
        cost_per_output_token=0.6/1000000
    )
    
    # Run product extraction
    print("\n🔍 Extracting product details...")
    extraction_results, effective_workers, extraction_time = lm.generate(extraction_inputs)
    llm_collector.add_inference_time(extraction_time)
    
    print(f"Extraction completed using {effective_workers} workers in {extraction_time:.4f}s")
    
    # Display extraction results
    print("\n📦 Product Extraction Results:")
    for i, (item, result) in enumerate(zip(available_items, extraction_results)):
        product_id = item.get('id', f'item_{i}')
        if 'error' in result:
            print(f"  {product_id}: Error - {result['error']}")
        else:
            print_product_details(result.get('product_details'), product_id)
    
    # Run evaluation if we have successful extractions
    successful_extractions = [(inp, res) for inp, res in zip(available_items, extraction_results) 
                            if 'error' not in res and res.get('product_details')]
    
    if successful_extractions:
        print(f"\n🔬 Evaluating {len(successful_extractions)} successful extractions...")
        
        # Format evaluation inputs
        evaluation_inputs = [
            ProductEvaluationPrompt().build(format_evaluation_record(inp, res))
            for inp, res in successful_extractions
        ]
        
        # Run evaluation
        evaluation_results, _, evaluation_time = lm.generate(evaluation_inputs)
        llm_collector.add_inference_time(evaluation_time)
        
        # Display evaluation results
        print("\n📊 Evaluation Results:")
        for i, (item, eval_result) in enumerate(zip([inp for inp, _ in successful_extractions], evaluation_results)):
            product_id = item.get('id', f'item_{i}')
            if 'error' in eval_result:
                print(f"  {product_id}: Evaluation error - {eval_result['error']}")
            else:
                evaluation = eval_result.get('evaluation', {})
                print(f"  {product_id}:")
                print(f"    Overall Quality: {evaluation.get('overall_quality', 'N/A')}")
                print(f"    Correct ID: {evaluation.get('correct_identification', 'N/A')}")
                print(f"    Accurate Details: {evaluation.get('accurate_details', 'N/A')}")
                print(f"    Complete: {evaluation.get('complete_extraction', 'N/A')}")
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
            schema_collector = SchemaMetricsCollector(ProductEvaluation, "evaluation_quality")
            schema_metrics = schema_collector.collect_metrics(evaluation_data)
            
            print("\n" + "=" * 60)
            print("🎯 EXTRACTION QUALITY METRICS")
            print("=" * 60)
            print(schema_collector.format_summary(schema_metrics))
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"Total items processed: {len(available_items)}")
    print(f"Successful extractions: {len(successful_extractions)}")
    print(f"Success rate: {len(successful_extractions)/len(available_items)*100:.1f}%")
    print(f"Total LLM calls: {llm_metrics['total_items']}")
    print(f"Total cost: ${llm_metrics['total_cost']:.6f}")
    print(f"Total time: {llm_metrics['total_inference_time']:.4f}s")
    
    print("\n✅ Vision extraction analysis completed!")
    
    return {
        "extraction_results": extraction_results,
        "evaluation_results": evaluation_results,
        "llm_metrics": llm_metrics
    }

if __name__ == "__main__":
    main()
