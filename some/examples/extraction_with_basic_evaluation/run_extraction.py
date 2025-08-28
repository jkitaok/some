"""
Generic Product Extraction with Basic Evaluation

This demonstrates structured data extraction with basic evaluation using the SOME library.
Uses the premade BasicEvaluation for quality assessment without ground truth data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

from some.inference import get_language_model
from some.metrics import LLMMetricsCollector
from some.io import read_json
from some.premade.extraction_evaluation import BasicEvaluation, EvaluationPrompt
from .my_prompt import ProductPrompt


def load_dataset(dataset_path: str = "dataset.json") -> List[Dict[str, Any]]:
    """Load the dataset from JSON file."""
    current_dir = Path(__file__).parent
    full_path = current_dir / dataset_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {full_path}")
    
    return read_json(str(full_path))


def run_basic_evaluation(inputs: List[Dict[str, Any]], results: List[Dict[str, Any]], lm, llm_collector: LLMMetricsCollector):
    """Run basic evaluation using the premade BasicEvaluation structure."""
    print("\n🔍 Running Basic Evaluation")
    print("=" * 50)
    
    # Format records for basic evaluation
    evaluation_records = []
    for (input_data, result_data) in zip(inputs, results):
        if result_data.get("product"):
            evaluation_records.append({
                "original_text": input_data["prompt_text"],
                "extraction_prompt": "Extract ProductSpec as JSON from this text and adhere strictly to the schema.",
                "expected_schema": input_data["response_format"].model_json_schema(),
                "extraction_output": result_data["product"]
            })
    
    if not evaluation_records:
        print("❌ No successful extractions to evaluate")
        return [], []
    
    # Build evaluation prompts using the premade EvaluationPrompt
    evaluation_inputs = [EvaluationPrompt().build(record) for record in evaluation_records]
    
    # Run evaluations with automatic metrics collection
    evaluation_results, _, eval_time = lm.generate(evaluation_inputs, metrics_collector=llm_collector)
    
    print(f"Basic Evaluation Results ({len(evaluation_records)} items):")
    evaluation_data = []
    
    correct_count = 0
    formatted_count = 0
    
    for i, r in enumerate(evaluation_results):
        eval_result = r.get('evaluation_result')
        if eval_result:
            evaluation_data.append(eval_result)
            
            correct = eval_result.get('correct', False)
            formatted = eval_result.get('formatted', False)
            
            if correct:
                correct_count += 1
            if formatted:
                formatted_count += 1
            
            print(f"  Item {i+1}: Correct={'✅' if correct else '❌'}, Formatted={'✅' if formatted else '❌'}")
            
            # Show reasoning if available
            reasoning = eval_result.get('reasoning', '')
            if reasoning:
                print(f"    Reasoning: {reasoning[:100]}...")
        else:
            print(f"  Item {i+1}: {r.get('error', 'No result')}")
    
    # Summary statistics
    if evaluation_data:
        print(f"\n📊 Basic Evaluation Summary:")
        print(f"  Correct: {correct_count}/{len(evaluation_data)} ({correct_count/len(evaluation_data)*100:.1f}%)")
        print(f"  Formatted: {formatted_count}/{len(evaluation_data)} ({formatted_count/len(evaluation_data)*100:.1f}%)")
    
    return evaluation_results, evaluation_data


def main(dataset_path: str = "dataset.json"):
    """Main function for running the extraction with basic evaluation."""
    print("🚀 Generic Product Extraction with Basic Evaluation")
    print("=" * 60)
    
    # Load dataset from JSON file
    try:
        dataset = load_dataset(dataset_path)
        print(f"📋 Loaded {len(dataset)} items from {dataset_path}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the dataset.json file exists in the same directory")
        return None
    
    # Extract text items for processing
    items = [{"text": item["text"]} for item in dataset]
    
    # Build inputs using the prompt builder
    inputs = [ProductPrompt().build(x) for x in items]
    
    # Get language model
    provider = "openai"
    model = "gpt-4o-mini"
    
    print(f"Using {provider} language model")
    lm = get_language_model(provider=provider, model=model)
    
    # Setup metrics collection
    llm_collector = LLMMetricsCollector(
        name="Product_Extraction_with_BasicEval",
        cost_per_input_token=0.15/1000000,   # GPT-4o-mini pricing
        cost_per_output_token=0.6/1000000
    )
    
    # Run extraction with automatic metrics collection
    results, effective_workers, extraction_time = lm.generate(inputs, metrics_collector=llm_collector)
    
    print(f"Extraction completed using {effective_workers} workers in {extraction_time:.4f}s")
    
    # Display extraction results
    print(f"\n📊 Extraction Results ({len(results)} items):")
    successful_extractions = 0
    for i, r in enumerate(results):
        product = r.get('product')
        if product:
            successful_extractions += 1
            print(f"  Item {i+1} ({dataset[i]['id']}): {product}")
        else:
            print(f"  Item {i+1} ({dataset[i]['id']}): ❌ {r.get('error', 'No result')}")
    
    print(f"\n✅ Successful extractions: {successful_extractions}/{len(results)}")
    
    # Run basic evaluation
    evaluation_results, evaluation_data = run_basic_evaluation(inputs, results, lm, llm_collector)
    
    # Collect LLM performance metrics
    llm_metrics = llm_collector.collect_metrics(results + evaluation_results)
    
    print("\n" + "=" * 50)
    print("📈 LLM PERFORMANCE METRICS")
    print("=" * 50)
    print(llm_collector.format_summary(llm_metrics))

    print("\n" + "=" * 50)
    print("💡 SUMMARY")
    print("=" * 50)
    print(f"Dataset items: {len(dataset)}")
    print(f"Successful extractions: {successful_extractions}/{len(results)} ({successful_extractions/len(results)*100:.1f}%)")
    print(f"Total LLM calls: {llm_metrics['total_items']}")
    print(f"Total cost: ${llm_metrics['total_cost']:.6f}")
    print(f"Total time: {llm_metrics['total_inference_time']:.2f}s")
    print(f"Evaluations completed: {len(evaluation_data)}")

    print("\n🎉 Analysis completed successfully!")
    
    return {
        "dataset": dataset,
        "extraction_results": results,
        "evaluation_results": evaluation_results,
        "llm_metrics": llm_metrics,
        "successful_extractions": successful_extractions
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run product extraction with basic evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset.json",
        help="Path to the JSON dataset file (default: dataset.json)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print(f"🔧 Configuration:")
    print(f"  Dataset: {args.dataset}")
    print()
    
    # Run extraction with basic evaluation
    results = main(dataset_path=args.dataset)
    
    if results:
        print(f"\n🎯 Final Summary:")
        print(f"  Processed {len(results['dataset'])} items from dataset")
        print(f"  Extraction success rate: {results['successful_extractions']}/{len(results['dataset'])}")
        print(f"  Total cost: ${results['llm_metrics']['total_cost']:.6f}")
        print(f"  Total time: {results['llm_metrics']['total_inference_time']:.2f}s")
    else:
        print("❌ Execution failed. Please check the error messages above.")
