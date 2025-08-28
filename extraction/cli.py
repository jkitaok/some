from __future__ import annotations

import sys
from typing import List

# CLI functionality has been removed - this package is now library-only
# Use the extraction modules directly in your Python code instead

def main(argv: List[str] | None = None):
    """CLI functionality has been removed. Use extraction modules directly in Python code."""
    print("CLI functionality has been removed from this package.")
    print("Please use the extraction modules directly in your Python code.")
    print("Example:")
    print("  from extraction.inference import get_language_model")
    print("  from extraction.prompting import BasePromptBuilder")
    sys.exit(1)


if __name__ == "__main__":
    main()
