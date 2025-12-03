# Experiment 01: Verifier Mechanics Analysis

## 1. Introduction: The Deterministic Judge

In the Axiom-RL framework, the Verifier is the source of truth. Unlike RLHF, which relies on a learned Reward Model (RM) that approximates human preference, our Verifier provides a **deterministic, objective signal**.

This document analyzes the current implementation in `axiom/verifier` and outlines the necessary upgrades to support the tiered reward system described in the Technical Audit.

## 2. Current Implementation (`axiom/verifier`)

The current Verifier is designed for **Binary Verification** (Pass/Fail), primarily used for Rejection Sampling and SFT data collection.

### 2.1 The Test Harness (`harness.py`)
The `TestHarness` class is the orchestrator.
*   **Input:** Solution Code (string), Problem Object.
*   **Process:**
    1.  Injects the solution code into a wrapper script (`_build_test_script`).
    2.  Injects test cases as a JSON payload.
    3.  Executes the script via `PythonSandbox`.
    4.  Parses the JSON output from `stdout`.
*   **Output:** `VerificationResult` (PASSED/FAILED, pass rate, error logs).

### 2.2 The Sandbox (`sandbox.py`)
The `PythonSandbox` provides a lightweight isolation layer.
*   **Mechanism:** `subprocess.run` with a restricted environment.
*   **Security:** Basic (no internet access, timeout enforcement). *Note: Not suitable for untrusted code in production.*
*   **Determinism:** Sets `PYTHONHASHSEED=0` to ensure consistent dictionary iteration order.

## 3. Gap Analysis: From SFT to GRPO

To support Group Relative Policy Optimization (GRPO), the Verifier must evolve from a **Binary Judge** to a **Scalar Reward Engine**.

### 3.1 Missing Tiered Rewards
The Technical Audit describes a 3-tier reward system. Here is the current status:

| Tier | Description | Current Status | Required Action |
| :--- | :--- | :--- | :--- |
| **1. Format** | Enforce `<think>` tags. | **Missing** | Implement Regex check in Trainer or Verifier wrapper. |
| **2. Accuracy** | Correctness of answer. | **Implemented** | `TestHarness` handles this (Pass/Fail). |
| **3. Safety** | Language/Content checks. | **Missing** | Add `langdetect` and keyword filters. |

### 3.2 The Reward Function
Currently, there is no function that converts a `VerificationResult` into a scalar reward (e.g., `1.0` or `-1.0`).

**Proposed Implementation:**
We need a new component, likely `RewardEngine`, that wraps the `TestHarness`:

```python
def calculate_reward(code: str, thinking: str, result: VerificationResult) -> float:
    reward = 0.0
    
    # Tier 1: Format
    if not has_think_tags(thinking):
        return -1.0  # Hard penalty
        
    # Tier 2: Accuracy
    if result.status == VerificationStatus.PASSED:
        reward += 1.0
    else:
        # Optional: Partial credit for passing some tests?
        # reward += result.pass_rate * 0.5 
        pass
        
    # Tier 3: Safety
    if detect_language(thinking) != "en":
        reward -= 0.5
        
    return reward
```

## 4. Conclusion

The `axiom/verifier` module provides a solid foundation for **correctness checking**. However, it is currently "RL-agnostic." It tells you *if* the code works, but it doesn't quantify *how good* the attempt was in the context of an RL optimization landscape.

**Next Steps for Research:**
1.  **Implement the Reward Function:** Create the logic to map verification results to scalar rewards.
2.  **Add Format Verification:** Ensure the RL agent is penalized for dropping the `<think>` tags.
3.  **Integrate with Trainer:** The future GRPO Trainer will need to call this Verifier inside its inner loop (64 times per step). Performance optimization (parallel execution) will be critical.
