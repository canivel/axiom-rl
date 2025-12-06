"""Curriculum Learning for GRPO Training.

Implements progressive difficulty training:
1. Start with easy problems where model succeeds
2. Gradually increase difficulty as model improves
3. Focus on problems with ~50-80% success rate (optimal learning zone)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum
import random


class CurriculumStrategy(Enum):
    """Curriculum progression strategies."""
    LINEAR = "linear"           # Fixed schedule: easy → medium → hard
    ADAPTIVE = "adaptive"       # Adjust based on performance
    SELF_PACED = "self_paced"   # Model chooses difficulty


@dataclass
class DifficultyLevel:
    """Configuration for a difficulty level."""
    name: str
    difficulty_range: tuple[int, int]  # (min, max) difficulty
    target_success_rate: float = 0.7   # Advance when reaching this
    min_samples: int = 10              # Min samples before advancing


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    # Difficulty levels (ordered easy to hard)
    levels: List[DifficultyLevel] = field(default_factory=lambda: [
        DifficultyLevel("easy", (1, 3), target_success_rate=0.8, min_samples=5),
        DifficultyLevel("medium", (4, 6), target_success_rate=0.7, min_samples=10),
        DifficultyLevel("hard", (7, 10), target_success_rate=0.6, min_samples=15),
    ])

    # Strategy
    strategy: CurriculumStrategy = CurriculumStrategy.ADAPTIVE

    # Adaptive settings
    success_window: int = 10  # Window for computing success rate
    advancement_threshold: float = 0.7  # Success rate to advance
    regression_threshold: float = 0.3   # Success rate to go back

    # Problem selection
    problem_types: List[str] = field(default_factory=list)  # Empty = all types
    problems_per_step: int = 1

    # Training settings
    steps_per_level: int = 20  # For LINEAR strategy
    max_total_steps: int = 100


class CurriculumScheduler:
    """Manages curriculum progression during GRPO training."""

    def __init__(
        self,
        config: CurriculumConfig,
        generators: Dict,  # problem_type -> generator
        seed: Optional[int] = None
    ):
        self.config = config
        self.generators = generators
        self.rng = random.Random(seed)

        # State
        self.current_level_idx = 0
        self.step_count = 0
        self.level_step_count = 0

        # History for adaptive scheduling
        self.success_history: List[float] = []
        self.level_history: List[int] = []

        # Filter generators by problem types
        if config.problem_types:
            self.active_generators = {
                k: v for k, v in generators.items()
                if k in config.problem_types
            }
        else:
            self.active_generators = generators

        if not self.active_generators:
            raise ValueError("No generators available for curriculum")

    @property
    def current_level(self) -> DifficultyLevel:
        """Get current difficulty level."""
        return self.config.levels[self.current_level_idx]

    @property
    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return self.step_count >= self.config.max_total_steps

    def get_problem(self):
        """Get next problem based on curriculum."""
        # Select random generator from active set
        gen_name = self.rng.choice(list(self.active_generators.keys()))
        generator = self.active_generators[gen_name]

        # Select difficulty from current level range
        min_diff, max_diff = self.current_level.difficulty_range
        difficulty = self.rng.randint(min_diff, max_diff)

        # Generate problem
        problem = generator.generate(difficulty=difficulty, num_test_cases=5)

        return problem, gen_name, difficulty

    def record_result(self, success_rate: float):
        """Record training step result and potentially advance/regress."""
        self.success_history.append(success_rate)
        self.level_history.append(self.current_level_idx)
        self.step_count += 1
        self.level_step_count += 1

        # Check for level change based on strategy
        if self.config.strategy == CurriculumStrategy.LINEAR:
            self._update_linear()
        elif self.config.strategy == CurriculumStrategy.ADAPTIVE:
            self._update_adaptive()
        elif self.config.strategy == CurriculumStrategy.SELF_PACED:
            self._update_self_paced(success_rate)

    def _update_linear(self):
        """Linear schedule: advance after fixed steps."""
        if self.level_step_count >= self.config.steps_per_level:
            self._advance_level()

    def _update_adaptive(self):
        """Adaptive schedule: advance/regress based on performance."""
        if len(self.success_history) < self.config.success_window:
            return

        # Compute recent success rate
        recent = self.success_history[-self.config.success_window:]
        avg_success = sum(recent) / len(recent)

        # Check for advancement
        if avg_success >= self.config.advancement_threshold:
            if self.level_step_count >= self.current_level.min_samples:
                self._advance_level()

        # Check for regression (only if not at first level)
        elif avg_success < self.config.regression_threshold:
            if self.current_level_idx > 0:
                self._regress_level()

    def _update_self_paced(self, success_rate: float):
        """Self-paced: stay in optimal learning zone (50-80% success)."""
        if success_rate > 0.85:
            self._advance_level()
        elif success_rate < 0.3 and self.current_level_idx > 0:
            self._regress_level()

    def _advance_level(self):
        """Move to next difficulty level."""
        if self.current_level_idx < len(self.config.levels) - 1:
            self.current_level_idx += 1
            self.level_step_count = 0
            print(f"[Curriculum] Advanced to level: {self.current_level.name}")

    def _regress_level(self):
        """Move to previous difficulty level."""
        if self.current_level_idx > 0:
            self.current_level_idx -= 1
            self.level_step_count = 0
            print(f"[Curriculum] Regressed to level: {self.current_level.name}")

    def get_stats(self) -> Dict:
        """Get curriculum statistics."""
        return {
            "current_level": self.current_level.name,
            "current_level_idx": self.current_level_idx,
            "total_steps": self.step_count,
            "level_steps": self.level_step_count,
            "avg_success": sum(self.success_history[-10:]) / max(len(self.success_history[-10:]), 1),
            "total_levels": len(self.config.levels),
        }

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get distribution of difficulties trained on."""
        dist = {}
        for level_idx in self.level_history:
            level = self.config.levels[level_idx]
            dist[level.name] = dist.get(level.name, 0) + 1
        return dist


def create_default_curriculum(
    generators: Dict,
    strategy: str = "adaptive",
    problem_types: Optional[List[str]] = None,
    max_steps: int = 50
) -> CurriculumScheduler:
    """Create a curriculum scheduler with default settings.

    Args:
        generators: Dict of problem_type -> generator
        strategy: "linear", "adaptive", or "self_paced"
        problem_types: List of problem types to include (None = all)
        max_steps: Maximum training steps

    Returns:
        Configured CurriculumScheduler
    """
    config = CurriculumConfig(
        strategy=CurriculumStrategy(strategy),
        problem_types=problem_types or [],
        max_total_steps=max_steps,
    )

    return CurriculumScheduler(config, generators)
