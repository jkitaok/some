"""
Integration Example: Using Labeled Data Evaluation in Your Pipeline

This example demonstrates how to integrate the labeled data evaluation system
into your own extraction pipeline for quality assessment and monitoring.
"""
from __future__ import annotations

from typing import Dict, Any, List
from pathlib import Path

# Import the labeled evaluation components
from some.premade.extraction_evaluation import (
    LabeledEvaluationPrompt,
    load_labeled_data,
    format_evaluation_record,
    calculate_accuracy_metrics,
    analyze_field_differences
)
from some.inference import get_language_model
from some.metrics import LLMMetricsCollector


def your_extraction_pipeline(text: str) -> Dict[str, Any]:
    """
    Placeholder for your actual extraction pipeline.
    
    Replace this with your real extraction logic that processes
    text and returns structured data.
    
    Args:
        text: Input text to extract information from
        
    Returns:
        Extracted structured data
    """
    # This is where you would call your actual extraction system
    # For example:
    # - Build extraction prompt
    # - Call language model
    # - Parse and validate results
    
    # Placeholder implementation
    return {
        "name": "Extracted Product Name",
        "price": 99.99,
        "features": ["feature1", "feature2"]
    }


def evaluate_extraction_quality(
    labeled_data_path: str,
    extraction_pipeline_func,
    model_provider: str = "openai",
    model_name: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    Evaluate your extraction pipeline against labeled ground truth data.
    
    Args:
        labeled_data_path: Path to your labeled evaluation data
        extraction_pipeline_func: Your extraction function
        model_provider: LLM provider for evaluation
        model_name: LLM model for evaluation
        
    Returns:
        Comprehensive evaluation results
    """
    print("🔍 Evaluating Extraction Pipeline Quality")
    print("=" * 50)
    
    # Load labeled data
    labeled_data = load_labeled_data(labeled_data_path)
    print(f"📋 Loaded {len(labeled_data)} labeled test cases")
    
    # Run your extraction pipeline on each test case
    print("⚡ Running extraction pipeline...")
    extraction_results = []
    for i, item in enumerate(labeled_data):
        print(f"   Processing item {i+1}/{len(labeled_data)}: {item['id']}")
        result = extraction_pipeline_func(item["original_text"])
        extraction_results.append(result)
    
    # Format for evaluation
    evaluation_records = [
        format_evaluation_record(item, result)
        for item, result in zip(labeled_data, extraction_results)
    ]
    
    # Build evaluation prompts
    prompt_builder = LabeledEvaluationPrompt()
    inputs = [prompt_builder.build(record) for record in evaluation_records]
    
    # Run evaluations
    print("🤖 Running quality evaluations...")
    lm = get_language_model(provider=model_provider, model=model_name)
    
    # Setup metrics collection
    llm_collector = LLMMetricsCollector(
        name="pipeline_evaluation",
        cost_per_input_token=2.5/1000000,   # GPT-4o pricing
        cost_per_output_token=10/1000000
    )
    
    results, _, evaluation_time = lm.generate(inputs)
    llm_collector.add_inference_time(evaluation_time)
    
    # Process results
    evaluations = []
    for result in results:
        if not result.get("error"):
            evaluation = result.get("labeled_evaluation_result")
            if evaluation:
                evaluations.append(evaluation)
    
    # Calculate aggregate metrics
    if evaluations:
        exact_matches = sum(1 for e in evaluations if e.get('exact_match', False))
        partial_matches = sum(1 for e in evaluations if e.get('partial_match', False))
        avg_accuracy = sum(e.get('accuracy_score', 0) for e in evaluations) / len(evaluations)
        
        print(f"\n📊 PIPELINE EVALUATION RESULTS")
        print("=" * 50)
        print(f"Test Cases: {len(labeled_data)}")
        print(f"Successful Evaluations: {len(evaluations)}")
        print(f"Exact Matches: {exact_matches}/{len(evaluations)} ({exact_matches/len(evaluations)*100:.1f}%)")
        print(f"Partial Matches: {partial_matches}/{len(evaluations)} ({partial_matches/len(evaluations)*100:.1f}%)")
        print(f"Average Accuracy Score: {avg_accuracy:.3f}")
        
        # LLM metrics
        llm_metrics = llm_collector.collect_metrics(results)
        print(f"\nEvaluation Cost: ${llm_metrics.get('total_cost', 0):.4f}")
        print(f"Evaluation Time: {evaluation_time:.2f}s")
    
    return {
        "labeled_data": labeled_data,
        "extraction_results": extraction_results,
        "evaluations": evaluations,
        "aggregate_metrics": {
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "average_accuracy": avg_accuracy,
            "total_test_cases": len(labeled_data)
        } if evaluations else None
    }


def continuous_quality_monitoring(
    labeled_data_path: str,
    extraction_pipeline_func,
    quality_threshold: float = 0.8
) -> bool:
    """
    Continuous quality monitoring for production systems.
    
    Args:
        labeled_data_path: Path to test data
        extraction_pipeline_func: Your extraction function
        quality_threshold: Minimum acceptable accuracy score
        
    Returns:
        True if quality meets threshold, False otherwise
    """
    print("🔄 Running Continuous Quality Check")
    print("=" * 40)
    
    results = evaluate_extraction_quality(labeled_data_path, extraction_pipeline_func)
    
    if results["aggregate_metrics"]:
        avg_accuracy = results["aggregate_metrics"]["average_accuracy"]
        
        if avg_accuracy >= quality_threshold:
            print(f"✅ Quality check PASSED: {avg_accuracy:.3f} >= {quality_threshold}")
            return True
        else:
            print(f"❌ Quality check FAILED: {avg_accuracy:.3f} < {quality_threshold}")
            print("🚨 Consider reviewing your extraction pipeline!")
            return False
    else:
        print("❌ Quality check FAILED: No successful evaluations")
        return False


def compare_extraction_models(
    labeled_data_path: str,
    model_configs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare different extraction models/configurations.
    
    Args:
        labeled_data_path: Path to test data
        model_configs: List of model configurations to compare
        
    Returns:
        Comparison results
    """
    print("⚖️  Comparing Extraction Models")
    print("=" * 40)
    
    comparison_results = {}
    
    for config in model_configs:
        model_name = config["name"]
        extraction_func = config["extraction_function"]
        
        print(f"\n🧪 Testing {model_name}...")
        results = evaluate_extraction_quality(labeled_data_path, extraction_func)
        
        if results["aggregate_metrics"]:
            comparison_results[model_name] = results["aggregate_metrics"]
    
    # Display comparison
    print(f"\n📊 MODEL COMPARISON RESULTS")
    print("=" * 50)
    
    for model_name, metrics in comparison_results.items():
        accuracy = metrics["average_accuracy"]
        exact_matches = metrics["exact_matches"]
        total_cases = metrics["total_test_cases"]
        
        print(f"{model_name}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Exact Matches: {exact_matches}/{total_cases}")
    
    return comparison_results


if __name__ == "__main__":
    # Example usage
    sample_data_path = "sample_labeled_data.json"
    
    # Single evaluation
    results = evaluate_extraction_quality(sample_data_path, your_extraction_pipeline)
    
    # Continuous monitoring
    quality_ok = continuous_quality_monitoring(sample_data_path, your_extraction_pipeline)
    
    # Model comparison (example)
    # model_configs = [
    #     {"name": "GPT-4o-mini", "extraction_function": your_gpt4_mini_pipeline},
    #     {"name": "GPT-4o", "extraction_function": your_gpt4_pipeline},
    # ]
    # comparison = compare_extraction_models(sample_data_path, model_configs)
