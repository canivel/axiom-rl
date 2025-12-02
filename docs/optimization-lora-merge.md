# Optimization: LoRA Merge vs PEFT Adapter Mode

## The Problem

During self-improvement iterations, we observed that inference with the trained PEFT model was **3-10x slower** than the base model. This significantly impacts experiment runtime:

- Iteration 0 (base model): ~5 minutes for 30 problems
- Iteration 1+ (PEFT adapter): ~15-30 minutes for 30 problems

## Understanding LoRA

### How LoRA (Low-Rank Adaptation) Works

LoRA adds small trainable "adapter" matrices to the model's attention layers without modifying the original weights:

```
Original forward pass:
    Output = Input × W

With LoRA:
    Output = Input × W + Input × (A × B)
             ↑ frozen    ↑ trainable (small matrices)
```

Where:
- `W` = Original weight matrix (e.g., 4096 × 4096 = 16M parameters)
- `A` = Low-rank matrix (e.g., 4096 × 16 = 65K parameters)
- `B` = Low-rank matrix (e.g., 16 × 4096 = 65K parameters)
- `A × B` = Reconstructed update (rank 16, much smaller than full matrix)

**Key insight:** Instead of training 16M parameters, we train 130K parameters (100x fewer).

## Two Inference Modes

### Mode 1: PEFT Adapter (Slow)

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE TIME                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input ──┬──→ [Base Model W] ────────────┐                 │
│           │                                │                 │
│           └──→ [LoRA Adapter A×B] ────────┼──→ ADD ──→ Output
│                                            │                 │
│   ⚠️  Two separate forward passes per layer!                │
│   ⚠️  Extra memory for adapter weights                      │
│   ⚠️  Overhead compounds across all attention layers        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**What happens during each forward pass:**
1. Compute `Input × W` (base model path)
2. Compute `Input × A` (first adapter multiplication)
3. Compute `(Input × A) × B` (second adapter multiplication)
4. Add results together

**Overhead per attention layer:**
- 2 extra matrix multiplications
- Memory for A and B matrices
- Synchronization between paths

**For Qwen 1.5B with 28 layers:**
- 28 × 2 = 56 extra matrix operations per forward pass
- This adds up significantly over thousands of tokens!

### Mode 2: Merged LoRA (Fast)

```
┌─────────────────────────────────────────────────────────────┐
│                    MERGE OPERATION (One-time)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   W_merged = W_original + (A × B)                           │
│                                                              │
│   This is just matrix addition! Done once after training.   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE TIME                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input ──→ [Merged Model W_merged] ──→ Output              │
│                                                              │
│   ✅ Single forward pass (same as original model!)          │
│   ✅ No extra memory for adapters                           │
│   ✅ Full inference speed                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**What happens:**
1. After training, compute `W_new = W + A×B` for each adapted layer
2. Replace original weights with merged weights
3. Discard A and B matrices
4. Model is now a standard model with no adapter overhead

## Performance Comparison

| Aspect | PEFT Adapter Mode | Merged LoRA |
|--------|-------------------|-------------|
| **Inference Speed** | 3-10x slower | Full speed (same as base) |
| **Memory (inference)** | Base + Adapter (~3.1GB) | Just merged (~3GB) |
| **Memory (merge operation)** | N/A | Negligible (in-place) |
| **Checkpoint Size** | ~4-40MB adapter only | ~3GB full model |
| **Flexibility** | Can swap adapters easily | Fixed weights |
| **Best For** | Multi-task, quick experiments | Production, iterative training |

## Why Merge is Better for Self-Improvement

### Current Flow (Slow)

```
Iteration 0:
  Base Model ──→ Train LoRA ──→ PEFT Model (adapter mode)
                                     │
Iteration 1:                         ▼
  PEFT Model (SLOW) ──→ Train more ──→ PEFT Model (SLOWER)
                                            │
Iteration 2:                                ▼
  PEFT Model (EVEN SLOWER) ──→ Train more ──→ ...
```

Each iteration, inference gets slower because:
1. Adapter overhead accumulates
2. More adapter weights to process
3. No benefit from previous training speed

### Optimized Flow (Fast)

```
Iteration 0:
  Base Model ──→ Train LoRA ──→ Merge ──→ Fast Merged Model
                                               │
Iteration 1:                                   ▼
  Merged Model (FAST) ──→ Train new LoRA ──→ Merge ──→ Fast Merged Model
                                                            │
Iteration 2:                                                ▼
  Merged Model (FAST) ──→ Train new LoRA ──→ Merge ──→ ...
```

Each iteration maintains full inference speed because:
1. LoRA is merged into weights after each training
2. Next iteration starts fresh with merged model
3. No adapter overhead during inference

## Implementation

### The Code Change

```python
# After training completes
trainer.train()

# OLD: Keep adapter mode (slow inference)
self.peft_model.eval()

# NEW: Merge and get fast inference
self.model = self.peft_model.merge_and_unload()
self.peft_model = None  # Ready for fresh LoRA next iteration
```

### What `merge_and_unload()` Does

1. **For each adapted layer:**
   - Retrieves original weight W
   - Retrieves LoRA matrices A and B
   - Computes: `W_merged = W + (A × B) × scaling_factor`
   - Replaces W with W_merged

2. **Cleanup:**
   - Removes all LoRA-specific modules
   - Removes adapter configuration
   - Returns a standard transformers model

3. **Memory:**
   - Frees adapter memory
   - Final model size = original model size
   - Merge operation is in-place (no extra memory spike)

## GPU Memory Analysis

For Qwen 1.5B model:

```
Before merge:
┌─────────────────────────────────────┐
│ Base Model (fp16):     ~3.0 GB     │
│ LoRA Adapters:         ~36 MB      │
│ Optimizer states:      ~72 MB      │ (if training)
│ ─────────────────────────────────── │
│ Total:                 ~3.1 GB     │
└─────────────────────────────────────┘

During merge (momentary):
┌─────────────────────────────────────┐
│ Base Model (fp16):     ~3.0 GB     │
│ LoRA Adapters:         ~36 MB      │
│ Merge computation:     ~negligible │ (matrix addition)
│ ─────────────────────────────────── │
│ Peak:                  ~3.1 GB     │
└─────────────────────────────────────┘

After merge:
┌─────────────────────────────────────┐
│ Merged Model (fp16):   ~3.0 GB     │
│ LoRA Adapters:         freed       │
│ ─────────────────────────────────── │
│ Total:                 ~3.0 GB     │
└─────────────────────────────────────┘
```

**Conclusion:** Merge is completely safe for any GPU that can run the base model.

## Expected Speedup

Based on our experiments:

| Phase | Before (PEFT) | After (Merged) | Speedup |
|-------|---------------|----------------|---------|
| Evaluation (18 problems) | ~15 min | ~5 min | 3x |
| Solution collection (18 problems) | ~15 min | ~5 min | 3x |
| Training | ~4 min | ~4 min | 1x (same) |
| **Total per iteration** | ~34 min | ~14 min | **2.4x** |

For a 10-iteration experiment:
- Before: ~340 minutes (~5.5 hours)
- After: ~140 minutes (~2.3 hours)
- **Saves 3+ hours per experiment!**

## Trade-offs

### What We Lose

1. **Adapter swapping:** Can't quickly switch between different LoRA adapters
   - Not needed for self-improvement (we train sequentially)

2. **Small checkpoints:** Must save full model (~3GB) instead of just adapter (~36MB)
   - We can still save adapters before merging if needed

3. **Reversibility:** Can't easily "undo" the LoRA training
   - Keep the original base model path for fresh starts

### What We Gain

1. **3x faster inference** after each training iteration
2. **Simpler model state** (just one model, not base + adapter)
3. **Lower memory during inference** (no adapter overhead)
4. **Consistent speed** across all iterations

## Conclusion

For iterative self-improvement experiments, **merging LoRA weights is the correct choice**:

- We don't need adapter flexibility (training is sequential)
- We do need fast inference (dominates runtime)
- Memory is not a concern (merge is lightweight)
- Speedup is significant (2-3x per iteration)

The implementation is a simple 2-line change that dramatically improves experiment throughput.

---

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation of Large Language Models
- [PEFT Library](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [merge_and_unload() docs](https://huggingface.co/docs/peft/main/en/package_reference/peft_model#peft.PeftModel.merge_and_unload)
