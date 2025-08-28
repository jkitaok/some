"""
Simple Product Data Extraction - Ready-to-Run Example

This script demonstrates a complete product data extraction pipeline using
the SOME library's building blocks. It includes sample data, extraction,
and metrics collection.

Run this example:
    python -m some.examples.premade.simple_product_extraction.run_simple_extraction
"""
from __future__ import annotations

import json
from typing import Dict, Any, List

from some.inference import get_language_model
from some.metrics import LLMMetricsCollector, SchemaMetricsCollector
from .schema import ProductData
from .prompt import ProductExtractionPrompt

def get_sample_data() -> List[Dict[str, Any]]:
    """
    Get sample product descriptions for extraction.
    
    Returns a variety of product types and description styles to demonstrate
    the extraction capabilities.
    """
    return [
        {
            "text": "The Apple iPhone 15 Pro Max features a 6.7-inch Super Retina XDR display, "
                   "A17 Pro chip, and titanium design. Starting at $1,199, it includes "
                   "a 48MP main camera, 5x telephoto zoom, and up to 1TB storage. "
                   "Available in Natural Titanium, Blue Titanium, White Titanium, and Black Titanium."
        },
        {
            "text": "Nike Air Max 270 running shoes offer maximum comfort with visible Air cushioning. "
                   "These lightweight sneakers feature breathable mesh upper and durable rubber outsole. "
                   "Price: $150. Available in multiple colorways. Perfect for daily training and casual wear. "
                   "Customer rating: 4.5/5 stars."
        },
        {
            "text": "Instant Pot Duo 7-in-1 Electric Pressure Cooker, 6 Quart. Functions as pressure cooker, "
                   "slow cooker, rice cooker, steamer, sauté pan, yogurt maker, and warmer. "
                   "Model: IP-DUO60. Currently on sale for $79.99 (regular price $99.99). "
                   "Over 50,000 5-star reviews. In stock and ready to ship."
        },
        {
            "text": "Samsung 65-inch QLED 4K Smart TV (QN65Q80C) delivers stunning picture quality "
                   "with Quantum Dot technology. Features include HDR10+, 120Hz refresh rate, "
                   "and built-in Alexa. Retail price: $1,299.99. Currently out of stock. "
                   "Expected restock date: next week."
        },
        {
            "text": "Levi's 501 Original Fit Jeans - the classic straight leg jean that started it all. "
                   "Made from 100% cotton denim. Available in various washes and sizes. "
                   "Price range: $59.50 - $69.50 depending on wash. Timeless style, "
                   "button fly, and signature arcuate stitching."
        }
    ]

def main():
    """
    Main function demonstrating the complete product extraction pipeline.
    
    This example shows:
    1. Sample data preparation
    2. Prompt building
    3. Language model inference
    4. Results processing
    5. Metrics collection and analysis
    """
    print("🚀 Simple Product Data Extraction Example")
    print("=" * 50)
    
    # Get sample product descriptions
    sample_items = get_sample_data()
    print(f"📝 Processing {len(sample_items)} product descriptions...")
    
    # Build prompts using the ProductExtractionPrompt
    prompt_builder = ProductExtractionPrompt()
    inputs = [prompt_builder.build(item) for item in sample_items]
    
    # Get language model (using OpenAI by default)
    # You can change this to other providers like "ollama" or custom providers
    provider = "openai"
    model = "gpt-4o-mini"  # Cost-effective model for this example
    
    print(f"🤖 Using {provider} language model: {model}")
    lm = get_language_model(provider=provider, model=model)
    
    # Setup metrics collection
    llm_collector = LLMMetricsCollector(
        name="simple_product_extraction",
        cost_per_input_token=0.15/1000000,   # GPT-4o-mini input pricing
        cost_per_output_token=0.6/1000000    # GPT-4o-mini output pricing
    )
    
    # Run extraction
    print("⚡ Running extraction...")
    results, effective_workers, extraction_time = lm.generate(inputs)
    llm_collector.add_inference_time(extraction_time)
    
    print(f"✅ Extraction completed using {effective_workers} workers in {extraction_time:.2f}s")
    
    # Display results
    print("\n📊 EXTRACTION RESULTS")
    print("=" * 50)
    
    extracted_products = []
    for i, result in enumerate(results):
        print(f"\n🏷️  Product {i+1}:")
        if result.get("error"):
            print(f"   ❌ Error: {result['error']}")
            continue
            
        product_data = result.get("product_data")
        if product_data:
            extracted_products.append(product_data)
            print(f"   Name: {product_data.get('name', 'N/A')}")
            print(f"   Price: ${product_data.get('price', 'N/A')}")
            print(f"   Brand: {product_data.get('brand', 'N/A')}")
            print(f"   Category: {product_data.get('category', 'N/A')}")
            if product_data.get('features'):
                print(f"   Features: {', '.join(product_data['features'][:3])}...")
            if product_data.get('rating'):
                print(f"   Rating: {product_data['rating']}/5 stars")
    
    # Collect and display metrics
    llm_metrics = llm_collector.collect_metrics(results)
    
    print(f"\n📈 PERFORMANCE METRICS")
    print("=" * 50)
    print(llm_collector.format_summary(llm_metrics))
    
    # Schema-based analysis of extracted data
    if extracted_products:
        schema_collector = SchemaMetricsCollector(ProductData, "product_analysis")
        schema_metrics = schema_collector.collect_metrics(extracted_products)
        
        print(f"\n🔍 DATA QUALITY ANALYSIS")
        print("=" * 50)
        print(schema_collector.format_summary(schema_metrics))
    
    # Summary insights
    print(f"\n💡 SUMMARY")
    print("=" * 50)
    print(f"Products processed: {len(sample_items)}")
    print(f"Successful extractions: {len(extracted_products)}")
    print(f"Success rate: {len(extracted_products)/len(sample_items)*100:.1f}%")
    print(f"Total cost: ${llm_metrics.get('total_cost', 0):.4f}")
    print(f"Average time per product: {extraction_time/len(sample_items):.2f}s")
    
    return {
        "extracted_products": extracted_products,
        "llm_metrics": llm_metrics,
        "schema_metrics": schema_metrics if extracted_products else None
    }

if __name__ == "__main__":
    main()
