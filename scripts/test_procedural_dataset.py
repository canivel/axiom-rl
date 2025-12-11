"""Test script for Procedural Dataset Integration."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.problems.dataset import ProblemDataset

def main():
    print("Testing Procedural Dataset Integration...")
    
    dataset = ProblemDataset()
    
    print("\nSampling 5 mixed problems (Difficulty 2):")
    problems = dataset.sample(5, difficulty=2)
    
    for p in problems:
        print(f"\n[{p.tags[0].upper()}] {p.title}")
        print(f"ID: {p.id}")
        print(f"Signature: {p.function_signature}")
        print(f"Test Case: {p.test_cases[0]}")
        
    print("\nTest passed!")

if __name__ == "__main__":
    main()
