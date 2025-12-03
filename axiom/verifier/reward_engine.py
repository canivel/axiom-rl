"""
Reward Engine for GRPO.

This module acts as the bridge between the GRPOTrainer and the TestHarness.
It converts model outputs (text) into scalar rewards (tensors) by:
1. Parsing the code from the reasoning trace.
2. Running the existing TestHarness.
3. Assigning scores (1.0 for pass, 0.0 for fail).
"""

import re
from typing import List
import torch

from .harness import TestHarness
from axiom.problems.dataset import ProblemDataset

class RewardEngine:
    """Calculates rewards for GRPO training."""
    
    def __init__(self, dataset: ProblemDataset):
        """
        Initialize the reward engine.
        
        Args:
            dataset: The problem dataset (needed to look up test cases)
        """
        self.harness = TestHarness()
        self.dataset = dataset
        
    def compute_rewards(self, prompts: List[str], completions: List[str]) -> torch.Tensor:
        """
        Compute rewards for a batch of completions.
        
        Args:
            prompts: List of prompt strings (used to extract problem_id)
            completions: List of completion strings (thinking + code)
            
        Returns:
            Tensor of rewards of shape (B,)
        """
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            # 1. Extract Problem ID from prompt
            # We assume the prompt contains "## Problem: Title" or we need a better way to track IDs.
            # For now, let's assume we can map prompt -> problem.
            # TODO: In a real training loop, we should pass problem_ids alongside prompts.
            # This is a limitation of the current Trainer signature.
            # FIX: We will parse the title from the prompt for now.
            problem = self._find_problem_from_prompt(prompt)
            
            if not problem:
                # If we can't find the problem, give 0 reward (safe fallback)
                rewards.append(0.0)
                continue
                
            # 2. Extract Code from completion
            code = self._extract_code(completion)
            
            if not code:
                # Format error penalty
                rewards.append(0.0) 
                continue
                
            # 3. Verify
            try:
                result = self.harness.verify(code, problem)
                score = 1.0 if result.passed_count == result.total_count else 0.0
                rewards.append(score)
            except Exception:
                # Runtime/System error
                rewards.append(0.0)
                
        return torch.tensor(rewards, dtype=torch.float32)

    def _extract_code(self, text: str) -> str:
        """Extract python code block from text."""
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _find_problem_from_prompt(self, prompt: str):
        """
        Identify the problem object from the prompt text.
        This is a heuristic since the standard Trainer doesn't pass metadata.
        """
        # The prompt format is: "## Problem: {title}\n"
        match = re.search(r"## Problem: (.*?)\n", prompt)
        if match:
            title = match.group(1).strip()
            # Search in dataset
            for p in self.dataset.problems:
                if p.title == title:
                    return p
        return None
