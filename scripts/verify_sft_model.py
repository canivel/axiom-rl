"""
Verify SFT Model.

This script loads a fine-tuned model and runs inference on a test problem
to verify that it has learned the correct reasoning format (<think> tags).
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Verify SFT Model Behavior")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base model")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter (e.g., models/test_sft)")
    parser.add_argument("--prompt", default="def two_sum(nums, target):", help="Test prompt")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading adapter: {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # Construct prompt with system instruction if needed, or raw
    # For Qwen, we usually use chat template, but let's try raw completion first
    # to see if it picked up the format from training.
    
    # We'll use a standard prompt format used in training
    full_prompt = f"""You are an expert Python programmer.
## Problem: Two Sum

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

## Function Signature
```python
{args.prompt}
```

## Required Format
<think>
...
</think>
```python
...
```"""

    print("\nGenerating...")
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
        
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("MODEL OUTPUT")
    print("="*50)
    print(output_text)
    print("="*50)
    
    # Verification Checks
    has_think_start = "<think>" in output_text
    has_think_end = "</think>" in output_text
    has_code = "```python" in output_text
    
    print("\nVERIFICATION RESULTS:")
    print(f"1. Has <think> tag: {'PASS' if has_think_start else 'FAIL'}")
    print(f"2. Has </think> tag: {'PASS' if has_think_end else 'FAIL'}")
    print(f"3. Has code block: {'PASS' if has_code else 'FAIL'}")
    
    if has_think_start and has_think_end and has_code:
        print("\n✅ SUCCESS: Model adopted the reasoning format.")
    else:
        print("\n❌ FAILURE: Model did not follow the format.")

if __name__ == "__main__":
    main()
