"""Test script for Procedural Generation."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.procedural import ArithmeticGenerator, RPNGenerator

def main():
    print("Testing Arithmetic Generator...")
    
    generator = ArithmeticGenerator()
    
    # Generate 5 problems of increasing difficulty
    for diff in range(1, 4):
        print(f"\n--- Difficulty {diff} ---")
        problems = generator.generate_batch(2, difficulty=diff)
        for p in problems:
            print(f"Problem: {p.title}")
            print(f"Solution: {p.solution_code.strip()}")
            
    print("\nTesting RPN Generator...")
    rpn_gen = RPNGenerator()
    for diff in range(1, 4):
        print(f"\n--- Difficulty {diff} ---")
        problems = rpn_gen.generate_batch(2, difficulty=diff)
        for p in problems:
            print(f"Problem: {p.title}")
            print(f"Solution: {p.solution_code.strip()}")

    print("\nTest passed!")

if __name__ == "__main__":
    main()
