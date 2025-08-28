"""
Simple Evaluation Demo - 2-4 Lines of Code Integration

This demonstrates how easy it is to integrate the premade evaluation system
into any extraction pipeline with just 2-4 lines of code.
"""

from some.premade.extraction_evaluation import BasicEvaluation, EvaluationPrompt
from some.inference import get_language_model

def evaluate_any_extraction(original_text, extraction_prompt, expected_schema, extraction_output):
    """
    Evaluate any extraction with just 2-4 lines of code!
    
    Args:
        original_text: The source text that was processed
        extraction_prompt: The prompt used for extraction  
        expected_schema: The schema definition (dict or string)
        extraction_output: The actual extraction result
        
    Returns:
        BasicEvaluation result with correct/formatted booleans
    """
    
    # Line 1: Create evaluation prompt
    evaluation_prompt = EvaluationPrompt()
    
    # Line 2: Build the evaluation input
    eval_input = evaluation_prompt.build({
        "original_text": original_text,
        "extraction_prompt": extraction_prompt,
        "expected_schema": expected_schema,
        "extraction_output": extraction_output
    })
    
    # Line 3: Get language model and run evaluation
    lm = get_language_model(provider="openai", model="gpt-4o-mini")
    results, _, _ = lm.generate([eval_input])
    
    # Line 4: Return the evaluation result
    return results[0].get("evaluation_result")

def demo():
    """Demo showing the 2-4 line integration in action."""
    
    print("🔍 Premade Evaluation Demo - 2-4 Lines of Code Integration")
    print("=" * 60)
    
    # Sample extraction scenario
    original_text = "The iPhone 15 Pro costs $999 and features a titanium design with USB-C port."
    extraction_prompt = "Extract product name and price"
    expected_schema = {"name": "string", "price": "number"}
    
    # Good extraction
    good_extraction = {"name": "iPhone 15 Pro", "price": 999}
    
    # Bad extraction  
    bad_extraction = {"name": "iPhone 15 Pro", "price": "$999"}  # Wrong type
    
    print("Testing GOOD extraction:")
    print(f"Input: {original_text}")
    print(f"Output: {good_extraction}")
    
    # Evaluate with just 2-4 lines!
    result1 = evaluate_any_extraction(original_text, extraction_prompt, expected_schema, good_extraction)
    print(f"✅ Result: Correct={result1.get('correct')}, Formatted={result1.get('formatted')}")
    
    print("\nTesting BAD extraction:")
    print(f"Input: {original_text}")
    print(f"Output: {bad_extraction}")
    
    # Evaluate with just 2-4 lines!
    result2 = evaluate_any_extraction(original_text, extraction_prompt, expected_schema, bad_extraction)
    print(f"❌ Result: Correct={result2.get('correct')}, Formatted={result2.get('formatted')}")
    
    print("\n💡 Integration Summary:")
    print("- Import: from some.premade.extraction_evaluation import BasicEvaluation, EvaluationPrompt")
    print("- Line 1: evaluation_prompt = EvaluationPrompt()")
    print("- Line 2: eval_input = evaluation_prompt.build({...})")
    print("- Line 3: results = lm.generate([eval_input])")
    print("- Line 4: return results[0].get('evaluation_result')")
    print("\nThat's it! 🎉")

if __name__ == "__main__":
    demo()
