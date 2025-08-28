"""
Example of using the some library for structured data extraction.

This demonstrates how to use the some package as a library without CLI functionality.
Run this as a Python module or import the functions for use in your own code.
"""
from __future__ import annotations

import os
from typing import Dict, Any

from some.inference import get_language_model
from some.metrics import LLMMetricsCollector, SchemaMetricsCollector
from .my_schema import BasicEvaluation
from .my_prompt import ProductPrompt, EvaluationPrompt
from .my_language_model import CustomLanguageModel  # Import triggers registration as "custom" provider

def format_evaluation_record(record):
    """Format a record for evaluation by combining input and output."""
    input_prompt = record[0]["messages"][0]["content"]
    expected_format = record[0]["response_format"].model_json_schema()
    extraction_output = record[1]["product"]

    return {
        "input_prompt": input_prompt,
        "expected_format": expected_format,
        "extraction_output": extraction_output
    }


def main():
    """Main function for running the example."""

    items = [
        {"text": "Widget X costs $19.99 and includes wifi, gps."},
        {"text": "Gadget Y is priced at $49.50, features: bluetooth, waterproofing"},
        {"text": "Device Z sells for $129.00 with premium materials and AI processing"},
    ]

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

    # Run extraction (timing automatically captured)
    results, effective_workers, extraction_time = lm.generate(inputs)
    llm_collector.add_inference_time(extraction_time)
    print(f"Extraction completed using {effective_workers} workers in {extraction_time:.4f}s")

    # Display extraction results
    print("\nExtraction Results:")
    for i, r in enumerate(results):
        print(f"{i}: {r.get('product')}")

    # Evaluation with automated metrics collection

    # Format records for evaluation
    formatted_records = [EvaluationPrompt().build(format_evaluation_record(r)) for r in zip(inputs, results)]

    # Run evaluation (timing automatically captured)
    evaluation_results, _, evaluation_time = lm.generate(formatted_records)
    llm_collector.add_inference_time(evaluation_time)

    # Display evaluation results
    print("\nEvaluation Results:")
    for i, r in enumerate(evaluation_results):
        print(f"{i}: {r.get('evaluation')}")

    # Collect LLM performance metrics
    all_llm_results = results + evaluation_results
    llm_metrics = llm_collector.collect_metrics(all_llm_results)

    print("\n" + "=" * 60, f"\nLLM PERFORMANCE METRICS\n", "=" * 60)
    print(llm_collector.format_summary(llm_metrics))

    # Collect schema-based data quality metrics for evaluations
    # Extract the evaluation data from the nested structure
    evaluation_data = []
    for r in evaluation_results:
        eval_data = r.get('evaluation')
        if eval_data is not None:
            evaluation_data.append(eval_data)

    schema_collector = SchemaMetricsCollector(BasicEvaluation, "evaluation_quality")
    schema_metrics = schema_collector.collect_metrics(evaluation_data)

    print("\n" + "=" * 60, f"\nDATA QUALITY METRICS\n", "=" * 60)
    print(schema_collector.format_summary(schema_metrics))

    print("\n" + "=" * 60, f"\nCOMBINED INSIGHTS\n", "=" * 60)
    print(f"Total LLM calls: {llm_metrics['total_items']}")
    print(f"Success rate: {llm_metrics['success_rate']:.1%}")
    print(f"Total cost: ${llm_metrics['total_cost']:.6f}")
    print(f"Total inference time: {llm_metrics['total_inference_time']:.4f}s")
    print(f"Evaluations analyzed: {schema_metrics['total_objects']}")

    if schema_metrics['total_objects'] > 0:
        evaluation_fields = schema_metrics['field_metrics']
        correct_field = evaluation_fields.get('correct', {})
        if 'true_percentage' in correct_field:
            print(f"Evaluation accuracy: {correct_field['true_percentage']:.1f}%")

    print("Analysis completed.")
    return {"llm_metrics": llm_metrics, "schema_metrics": schema_metrics}


if __name__ == "__main__":
    main()