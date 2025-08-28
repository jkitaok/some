"""
Example of using the some library for structured data extraction.

This demonstrates how to use the some package as a library without CLI functionality.
Run this as a Python module or import the functions for use in your own code.

KEY FEATURES:
- Simple JSON dataset with automatic evaluation
- Runs both manual and labeled evaluation when ground truth is available
- Clean, streamlined interface for easy adoption
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

from some.inference import get_language_model
from some.metrics import LLMMetricsCollector
from some.io import read_json
from some.premade.extraction_evaluation import (
    EvaluationPrompt,
    LabeledEvaluation,
    LabeledEvaluationPrompt
)
from .my_prompt import ProductPrompt
from .my_language_model import CustomLanguageModel  # Import triggers registration as "custom" provider

def load_dataset(dataset_path: str = "dataset.json") -> List[Dict[str, Any]]:
    """Load the dataset from JSON file."""
    current_dir = Path(__file__).parent
    full_path = current_dir / dataset_path

    if not full_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {full_path}")

    return read_json(str(full_path))


def run_evaluations(dataset: List[Dict[str, Any]], inputs: List[Dict[str, Any]], results: List[Dict[str, Any]], lm, llm_collector: LLMMetricsCollector):
    """Run both manual and labeled evaluations when applicable."""
    all_evaluation_results = []
    all_evaluation_data = []

    # Always run manual evaluation
    print("\n🔍 Running Manual Evaluation")
    print("=" * 50)

    # Format records for manual evaluation
    manual_records = []
    for (input_data, result_data) in zip(inputs, results):
        if result_data.get("product"):
            manual_records.append({
                "original_text": input_data["prompt_text"],
                "extraction_prompt": "Extract ProductSpec as JSON from this text and adhere strictly to the schema.",
                "expected_schema": input_data["response_format"].model_json_schema(),
                "extraction_output": result_data["product"]
            })

    if manual_records:
        manual_inputs = [EvaluationPrompt().build(record) for record in manual_records]
        manual_results, _, eval_time = lm.generate(manual_inputs, metrics_collector=llm_collector)

        print("Manual Evaluation Results:")
        for i, r in enumerate(manual_results):
            eval_result = r.get('evaluation_result')
            if eval_result:
                all_evaluation_data.append(eval_result)
                print(f"  Item {i+1}: Correct={eval_result.get('correct')}, Formatted={eval_result.get('formatted')}")
            else:
                print(f"  Item {i+1}: {r.get('error', 'No result')}")

        all_evaluation_results.extend(manual_results)

    # Run labeled evaluation if ground truth is available
    labeled_items = [item for item in dataset if "ground_truth" in item and item["ground_truth"]]
    if labeled_items:
        print(f"\n🏷️ Running Labeled Data Evaluation ({len(labeled_items)} items with ground truth)")
        print("=" * 50)

        # Create simple evaluation records - just extraction output vs ground truth
        labeled_records = []
        for item, result in zip(labeled_items, results[:len(labeled_items)]):
            extraction_output = result.get("product")
            if extraction_output:
                labeled_records.append({
                    "original_text": item["text"],
                    "extraction_prompt": "Extract ProductSpec as JSON from this text and adhere strictly to the schema.",
                    "expected_schema": {"name": "string", "price": "number", "features": "array of strings"},
                    "extraction_output": extraction_output,
                    "ground_truth": item["ground_truth"],
                    "evaluation_context": ""
                })

        if labeled_records:
            # Use the proper LabeledEvaluationPrompt builder
            prompt_builder = LabeledEvaluationPrompt()
            labeled_inputs = [prompt_builder.build(record) for record in labeled_records]

            labeled_results, _, eval_time = lm.generate(labeled_inputs, metrics_collector=llm_collector)

            print("Labeled Evaluation Results:")
            exact_matches = 0
            total_accuracy = 0

            for i, r in enumerate(labeled_results):
                eval_result = r.get('labeled_evaluation_result')
                if eval_result:
                    all_evaluation_data.append(eval_result)

                    exact_match = eval_result.get('exact_match', False)
                    accuracy = eval_result.get('accuracy_score', 0)

                    if exact_match:
                        exact_matches += 1
                    total_accuracy += accuracy

                    print(f"  Item {i+1} ({labeled_items[i]['id']}):")
                    print(f"    Exact Match: {'✅' if exact_match else '❌'}")
                    print(f"    Accuracy Score: {accuracy:.2f}")

                    # Show field issues if any
                    incorrect_count = eval_result.get('incorrect_fields_count', 0)
                    if incorrect_count > 0:
                        incorrect_list = eval_result.get('incorrect_fields_list', 'N/A')
                        print(f"    Incorrect Fields ({incorrect_count}): {incorrect_list}")
                else:
                    print(f"  Item {i+1}: {r.get('error', 'No result')}")

            # Summary
            if len(labeled_records) > 0:
                avg_accuracy = total_accuracy / len(labeled_records)
                print(f"\n📊 Labeled Evaluation Summary:")
                print(f"  Exact Matches: {exact_matches}/{len(labeled_records)} ({exact_matches/len(labeled_records)*100:.1f}%)")
                print(f"  Average Accuracy: {avg_accuracy:.3f}")

            all_evaluation_results.extend(labeled_results)

    return all_evaluation_results, all_evaluation_data


def main(dataset_path: str = "dataset.json"):
    """Main function for running the example."""
    print("🚀 Generic Product Extraction with Evaluation")
    print("=" * 50)

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
    provider = "openai"  # Change to "custom" if you want to test the mock model
    lm = get_language_model(provider="openai", model="gpt-5-nano")
    print(f"Using {provider} language model")

    # Setup LLM metrics collector (automatic timing from language model)
    llm_collector = LLMMetricsCollector(
        name="product_extraction",
        cost_per_input_token=0.05/1000000,  # GPT-4 pricing example
        cost_per_output_token=0.4/1000000
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

    # Run evaluations (both manual and labeled when applicable)
    all_evaluation_results, all_evaluation_data = run_evaluations(dataset, inputs, results, lm, llm_collector)

    # Collect LLM performance metrics
    all_llm_results = results + all_evaluation_results
    llm_metrics = llm_collector.collect_metrics(all_llm_results)

    print("\n" + "=" * 70)
    print("📈 LLM PERFORMANCE METRICS")
    print("=" * 70)
    print(llm_collector.format_summary(llm_metrics))

    print("\n" + "=" * 50)
    print("� SUMMARY")
    print("=" * 50)
    print(f"Dataset items: {len(dataset)}")
    print(f"Successful extractions: {successful_extractions}/{len(results)} ({successful_extractions/len(results)*100:.1f}%)")
    print(f"Total LLM calls: {llm_metrics['total_items']}")
    print(f"Total cost: ${llm_metrics['total_cost']:.6f}")
    print(f"Total time: {llm_metrics['total_inference_time']:.2f}s")
    print(f"Evaluations completed: {len(all_evaluation_data)}")

    print("\n🎉 Analysis completed successfully!")

    return {
        "dataset": dataset,
        "extraction_results": results,
        "evaluation_results": all_evaluation_results,
        "llm_metrics": llm_metrics,
        "successful_extractions": successful_extractions
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run generic product extraction with automatic evaluation"
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

    # Run extraction and evaluation
    results = main(dataset_path=args.dataset)

    if results:
        print(f"\n🎯 Final Summary:")
        print(f"  Processed {len(results['dataset'])} items from dataset")
        print(f"  Extraction success rate: {results['successful_extractions']}/{len(results['dataset'])}")
        print(f"  Total cost: ${results['llm_metrics']['total_cost']:.6f}")
        print(f"  Total time: {results['llm_metrics']['total_inference_time']:.2f}s")
    else:
        print("❌ Execution failed. Please check the error messages above.")