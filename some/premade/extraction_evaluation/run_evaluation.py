"""
Extraction Evaluation - Ready-to-Run Example

This script demonstrates how to evaluate extraction results using the SOME library.
It supports both manual evaluation and labeled data evaluation modes.
Includes sample extraction scenarios with both good and problematic outputs
to showcase the evaluation capabilities.

Run this example:
    python -m some.premade.extraction_evaluation.run_evaluation

    # Or with labeled data:
    python -m some.premade.extraction_evaluation.run_evaluation --labeled-data path/to/labeled_data.json
"""
from __future__ import annotations

import json
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path

from some.inference import get_language_model
from some.metrics import LLMMetricsCollector, SchemaMetricsCollector
from .schema import BasicEvaluation, LabeledEvaluation, EvaluationMode
from .prompt import EvaluationPrompt, LabeledEvaluationPrompt
from .labeled_utils import load_labeled_data, format_evaluation_record, calculate_accuracy_metrics, analyze_field_differences

def get_sample_evaluation_scenarios() -> List[Dict[str, Any]]:
    """
    Get sample extraction scenarios for evaluation.
    
    Returns various extraction results with different quality levels
    to demonstrate the evaluation system.
    """
    return [
        {
            "scenario_name": "Good Product Extraction",
            "original_text": "The Apple MacBook Pro 16-inch features the M3 Max chip, 36GB unified memory, "
                           "and 1TB SSD storage. Starting at $3,999, it includes a Liquid Retina XDR display "
                           "with 3456x2234 resolution and up to 22 hours of battery life. Available in Space Black and Silver.",
            "extraction_prompt": "Extract product information including name, price, features, and specifications.",
            "expected_schema": {
                "name": "string",
                "price": "number", 
                "features": "array of strings",
                "specifications": "object"
            },
            "extraction_output": {
                "name": "Apple MacBook Pro 16-inch",
                "price": 3999,
                "features": ["M3 Max chip", "36GB unified memory", "1TB SSD storage", "Liquid Retina XDR display", "22 hours battery life"],
                "specifications": {
                    "display_resolution": "3456x2234",
                    "colors": ["Space Black", "Silver"]
                }
            }
        },
        {
            "scenario_name": "Incomplete Extraction",
            "original_text": "Nike Air Jordan 1 Retro High OG in Chicago colorway. Features premium leather upper, "
                           "Air-Sole unit in heel, and rubber outsole. Originally released in 1985, "
                           "this iconic basketball shoe retails for $170. Size range: 7-18. Limited stock available.",
            "extraction_prompt": "Extract complete product information including all mentioned details.",
            "expected_schema": {
                "name": "string",
                "price": "number",
                "brand": "string", 
                "features": "array of strings",
                "availability": "string",
                "size_range": "string"
            },
            "extraction_output": {
                "name": "Air Jordan 1",
                "price": 170,
                "features": ["leather upper", "Air-Sole unit"]
                # Missing: brand, availability, size_range, colorway info
            }
        },
        {
            "scenario_name": "Format Violation",
            "original_text": "Samsung Galaxy S24 Ultra smartphone with 200MP camera, 6.8-inch Dynamic AMOLED display, "
                           "and S Pen support. Available in Titanium Gray, Titanium Black, Titanium Violet, "
                           "and Titanium Yellow. Price starts at $1,299.99 for 256GB model.",
            "extraction_prompt": "Extract smartphone specifications in structured format.",
            "expected_schema": {
                "name": "string",
                "price": "number",
                "display_size": "number", 
                "camera_mp": "number",
                "colors": "array of strings"
            },
            "extraction_output": {
                "name": "Samsung Galaxy S24 Ultra",
                "price": "$1,299.99",  # Should be number, not string
                "display_size": "6.8-inch",  # Should be number, not string with unit
                "camera_mp": "200MP camera",  # Should be number, not string
                "colors": "Titanium Gray, Titanium Black, Titanium Violet, Titanium Yellow"  # Should be array
            }
        },
        {
            "scenario_name": "Inaccurate Information",
            "original_text": "Tesla Model 3 Long Range offers up to 358 miles of EPA-estimated range. "
                           "It accelerates from 0-60 mph in 4.2 seconds and has a top speed of 145 mph. "
                           "Starting price is $47,740. Features include Autopilot, 15-inch touchscreen, "
                           "and over-the-air software updates.",
            "extraction_prompt": "Extract vehicle specifications and features accurately.",
            "expected_schema": {
                "model": "string",
                "range_miles": "number",
                "acceleration_0_60": "number",
                "top_speed": "number", 
                "price": "number",
                "features": "array of strings"
            },
            "extraction_output": {
                "model": "Tesla Model 3 Long Range",
                "range_miles": 400,  # Incorrect: should be 358
                "acceleration_0_60": 3.8,  # Incorrect: should be 4.2
                "top_speed": 155,  # Incorrect: should be 145
                "price": 47740,  # Correct
                "features": ["Autopilot", "15-inch touchscreen", "OTA updates", "Full Self-Driving"]  # Added feature not mentioned
            }
        }
    ]

def run_manual_evaluation():
    """
    Run manual evaluation without labeled data.

    This demonstrates the original evaluation functionality using sample scenarios.
    """
    print("🔍 Manual Extraction Evaluation")
    print("=" * 50)

    # Get sample evaluation scenarios
    scenarios = get_sample_evaluation_scenarios()
    print(f"📋 Evaluating {len(scenarios)} extraction scenarios...")

    # Build evaluation prompts
    prompt_builder = EvaluationPrompt()
    inputs = [prompt_builder.build(scenario) for scenario in scenarios]

    # Get language model for evaluation
    provider = "openai"
    model = "gpt-4o"  # Using more capable model for evaluation tasks

    print(f"🤖 Using {provider} language model: {model}")
    lm = get_language_model(provider=provider, model=model)

    # Setup metrics collection
    llm_collector = LLMMetricsCollector(
        name="manual_extraction_evaluation",
        cost_per_input_token=2.5/1000000,   # GPT-4o input pricing
        cost_per_output_token=10/1000000    # GPT-4o output pricing
    )

    # Run evaluations
    print("⚡ Running evaluations...")
    results, effective_workers, evaluation_time = lm.generate(inputs)
    llm_collector.add_inference_time(evaluation_time)

    print(f"✅ Evaluation completed using {effective_workers} workers in {evaluation_time:.2f}s")

    # Display results
    print("\n📊 MANUAL EVALUATION RESULTS")
    print("=" * 50)
    
    evaluations = []
    for i, (scenario, result) in enumerate(zip(scenarios, results)):
        print(f"\n🎯 Scenario {i+1}: {scenario['scenario_name']}")
        
        if result.get("error"):
            print(f"   ❌ Error: {result['error']}")
            continue
            
        evaluation = result.get("evaluation_result")
        if evaluation:
            evaluations.append(evaluation)
            print(f"   Correct: {'✅ Yes' if evaluation.get('correct') else '❌ No'}")
            print(f"   Formatted: {'✅ Yes' if evaluation.get('formatted') else '❌ No'}")
            if evaluation.get('confidence'):
                print(f"   Confidence: {evaluation.get('confidence'):.2f}")
            if evaluation.get('reasoning'):
                print(f"   Reasoning: {evaluation.get('reasoning')}")
    
    # Collect and display metrics
    llm_metrics = llm_collector.collect_metrics(results)
    
    print(f"\n📈 PERFORMANCE METRICS")
    print("=" * 50)
    print(llm_collector.format_summary(llm_metrics))
    
    # Schema-based analysis of evaluations
    if evaluations:
        schema_collector = SchemaMetricsCollector(BasicEvaluation, "evaluation_analysis")
        schema_metrics = schema_collector.collect_metrics(evaluations)

        print(f"\n🔍 EVALUATION QUALITY ANALYSIS")
        print("=" * 50)
        print(schema_collector.format_summary(schema_metrics))

    return results


def run_labeled_evaluation(labeled_data_path: str):
    """
    Run evaluation using labeled ground truth data.

    Args:
        labeled_data_path: Path to the labeled data file (JSON or JSONL)
    """
    print("🏷️ Labeled Data Extraction Evaluation")
    print("=" * 50)

    # Load labeled data
    try:
        labeled_data = load_labeled_data(labeled_data_path)
        print(f"📋 Loaded {len(labeled_data)} labeled evaluation items from {labeled_data_path}")
    except Exception as e:
        print(f"❌ Error loading labeled data: {e}")
        return None

    # For demonstration, we'll simulate extraction results
    # In a real scenario, you would run your extraction pipeline here
    print("⚠️  Note: Using simulated extraction results for demonstration")
    print("   In practice, you would run your extraction pipeline on the original_text")

    evaluation_scenarios = []
    for item in labeled_data:
        # Simulate an extraction result (in practice, this would come from your extraction pipeline)
        simulated_extraction = simulate_extraction_result(item)

        # Format for evaluation
        evaluation_record = format_evaluation_record(item, simulated_extraction)
        evaluation_scenarios.append(evaluation_record)

    # Build labeled evaluation prompts
    prompt_builder = LabeledEvaluationPrompt()
    inputs = [prompt_builder.build(scenario) for scenario in evaluation_scenarios]

    # Get language model for evaluation
    provider = "openai"
    model = "gpt-4o"  # Using more capable model for evaluation tasks

    print(f"🤖 Using {provider} language model: {model}")
    lm = get_language_model(provider=provider, model=model)

    # Setup metrics collection
    llm_collector = LLMMetricsCollector(
        name="labeled_extraction_evaluation",
        cost_per_input_token=2.5/1000000,   # GPT-4o input pricing
        cost_per_output_token=10/1000000    # GPT-4o output pricing
    )

    # Run evaluations
    print("⚡ Running labeled evaluations...")
    results, effective_workers, evaluation_time = lm.generate(inputs)
    llm_collector.add_inference_time(evaluation_time)

    print(f"✅ Evaluation completed using {effective_workers} workers in {evaluation_time:.2f}s")

    # Display results
    print("\n📊 LABELED EVALUATION RESULTS")
    print("=" * 50)

    evaluations = []
    for i, result in enumerate(results):
        if result.get("error"):
            print(f"❌ Evaluation {i+1} failed: {result['error']}")
            continue

        evaluation = result.get("labeled_evaluation_result")
        if evaluation:
            evaluations.append(evaluation)
            print(f"\n📝 Evaluation {i+1}:")
            print(f"   Exact Match: {'✅' if evaluation.get('exact_match') else '❌'}")
            print(f"   Partial Match: {'✅' if evaluation.get('partial_match') else '❌'}")
            print(f"   Formatted: {'✅' if evaluation.get('formatted') else '❌'}")
            print(f"   Accuracy Score: {evaluation.get('accuracy_score', 0):.2f}")

            # Display field analysis counts
            missing_count = evaluation.get('missing_fields_count', 0)
            extra_count = evaluation.get('extra_fields_count', 0)
            incorrect_count = evaluation.get('incorrect_fields_count', 0)

            if missing_count > 0:
                missing_list = evaluation.get('missing_fields_list', 'N/A')
                print(f"   Missing Fields ({missing_count}): {missing_list}")
            if extra_count > 0:
                extra_list = evaluation.get('extra_fields_list', 'N/A')
                print(f"   Extra Fields ({extra_count}): {extra_list}")
            if incorrect_count > 0:
                incorrect_list = evaluation.get('incorrect_fields_list', 'N/A')
                print(f"   Incorrect Fields ({incorrect_count}): {incorrect_list}")

            if evaluation.get('reasoning'):
                print(f"   Reasoning: {evaluation['reasoning'][:100]}...")

    # Collect and display metrics
    llm_metrics = llm_collector.collect_metrics(results)

    print(f"\n📈 PERFORMANCE METRICS")
    print("=" * 50)
    print(llm_collector.format_summary(llm_metrics))

    # Schema-based analysis of evaluations
    if evaluations:
        schema_collector = SchemaMetricsCollector(LabeledEvaluation, "labeled_evaluation_analysis")
        schema_metrics = schema_collector.collect_metrics(evaluations)

        print(f"\n🔍 LABELED EVALUATION QUALITY ANALYSIS")
        print("=" * 50)
        print(schema_collector.format_summary(schema_metrics))

        # Calculate aggregate metrics
        exact_matches = sum(1 for e in evaluations if e.get('exact_match', False))
        partial_matches = sum(1 for e in evaluations if e.get('partial_match', False))
        avg_accuracy = sum(e.get('accuracy_score', 0) for e in evaluations) / len(evaluations)

        print(f"\n📊 AGGREGATE METRICS")
        print("=" * 50)
        print(f"Exact Matches: {exact_matches}/{len(evaluations)} ({exact_matches/len(evaluations)*100:.1f}%)")
        print(f"Partial Matches: {partial_matches}/{len(evaluations)} ({partial_matches/len(evaluations)*100:.1f}%)")
        print(f"Average Accuracy Score: {avg_accuracy:.3f}")

    return results


def simulate_extraction_result(labeled_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate an extraction result for demonstration purposes.

    In a real scenario, this would be replaced by your actual extraction pipeline.

    Args:
        labeled_item: The labeled data item

    Returns:
        Simulated extraction result
    """
    ground_truth = labeled_item["ground_truth"]

    # Create a simulated extraction that's partially correct
    simulated = {}

    for key, value in ground_truth.items():
        if isinstance(value, str):
            # Sometimes get it right, sometimes wrong, sometimes missing
            import random
            rand = random.random()
            if rand < 0.6:  # 60% chance correct
                simulated[key] = value
            elif rand < 0.8:  # 20% chance wrong
                simulated[key] = f"incorrect_{value}"
            # 20% chance missing (don't add to simulated)
        elif isinstance(value, (int, float)):
            # Numeric values - sometimes slightly off
            import random
            if random.random() < 0.7:  # 70% chance correct
                simulated[key] = value
            else:  # 30% chance slightly off
                simulated[key] = value + random.randint(-2, 2)
        elif isinstance(value, list):
            # Lists - sometimes missing items
            import random
            if random.random() < 0.8:  # 80% chance to include field
                # Include 70-100% of items
                num_items = max(1, int(len(value) * random.uniform(0.7, 1.0)))
                simulated[key] = value[:num_items]
        else:
            # Other types - just copy
            simulated[key] = value

    return simulated

def main(labeled_data_path: Optional[str] = None):
    """
    Main function that supports both manual and labeled data evaluation modes.

    Args:
        labeled_data_path: Optional path to labeled data file. If provided, runs labeled evaluation.
                          If None, runs manual evaluation with sample scenarios.
    """
    if labeled_data_path:
        return run_labeled_evaluation(labeled_data_path)
    else:
        return run_manual_evaluation()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run extraction evaluation with optional labeled data support"
    )
    parser.add_argument(
        "--labeled-data",
        type=str,
        help="Path to labeled data file (JSON or JSONL format)"
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "labeled"],
        help="Evaluation mode (auto-detected based on --labeled-data if not specified)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine evaluation mode
    if args.labeled_data:
        if not Path(args.labeled_data).exists():
            print(f"❌ Error: Labeled data file not found: {args.labeled_data}")
            exit(1)
        main(labeled_data_path=args.labeled_data)
    else:
        if args.mode == "labeled":
            print("❌ Error: --labeled-data is required when using labeled mode")
            exit(1)
        main()
