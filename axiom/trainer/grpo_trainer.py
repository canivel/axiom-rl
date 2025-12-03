"""GRPO Trainer implementation."""

from pathlib import Path
from typing import Callable, Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

from .grpo_config import GRPOConfig
from .lora_config import get_lora_config

class GRPOTrainer:
    """
    Trainer for Group Relative Policy Optimization (GRPO).
    
    Implements the RL loop:
    1. Rollout: Generate G samples per prompt
    2. Evaluation: Score samples with Verifier
    3. Advantage: Compute group relative advantages
    4. Update: Optimize policy using GRPO loss
    """
    
    def __init__(
        self,
        config: GRPOConfig,
        reward_function: Callable[[List[str], List[str]], torch.Tensor],
        processing_class: Optional[Any] = None, # Tokenizer
    ):
        """
        Initialize the GRPO trainer.
        
        Args:
            config: GRPO configuration
            reward_function: Callable taking (prompts, completions) and returning rewards tensor
            processing_class: Tokenizer (optional, will load from config if None)
        """
        self.config = config
        self.reward_function = reward_function
        self.tokenizer = processing_class
        
        self.policy_model = None
        self.ref_model = None
        self.optimizer = None
        
    def setup(self):
        """Load models and prepare for training."""
        print(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer if needed
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        # Load Policy Model (Trainable)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map="auto",
            trust_remote_code=True
        )
        
        # Apply LoRA to Policy
        lora_config = get_lora_config(self.config)
        self.policy_model = get_peft_model(self.policy_model, lora_config)
        self.policy_model.print_trainable_parameters()
        
        # Load Reference Model (Frozen)
        # For MVP, we can use the same base model with disabled adapters or a separate copy
        # To save memory, we might offload this or use the same model with LoRA disabled
        print("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map="auto",
            trust_remote_code=True
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate
        )
        
    def train(self, train_dataset):
        """
        Run the GRPO training loop.
        
        Args:
            train_dataset: Dataset returning prompts
        """
        if self.policy_model is None:
            self.setup()
            
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        print("Starting GRPO Training...")
        
        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(dataloader):
                prompts = batch["prompts"]
                
                # 1. Rollout
                generations = self._rollout(prompts)
                
                # 2. Reward Calculation
                rewards = self._compute_rewards(prompts, generations)
                
                # 3. GRPO Update
                loss = self._compute_loss(prompts, generations, rewards)
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if step % self.config.logging_steps == 0:
                    print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Avg Reward: {rewards.mean().item():.4f}")

    def _rollout(self, prompts: List[str]) -> List[List[str]]:
        """
        Generate G completions for each prompt.
        
        Returns:
            List of lists, where outer list is batch, inner list is G generations.
        """
        self.policy_model.eval()
        
        # Set padding side to left for generation
        self.tokenizer.padding_side = "left"
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.policy_model.device)
        
        # Repeat inputs for G generations
        # Shape: (B * G, seq_len)
        G = self.config.num_generations
        input_ids = inputs.input_ids.repeat_interleave(G, dim=0)
        attention_mask = inputs.attention_mask.repeat_interleave(G, dim=0)
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_seq_length, # Generate up to max length
                do_sample=True,
                temperature=0.7, # TODO: Make configurable
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
        # Decode
        # outputs contains input_ids + generated_ids
        # We only want the generated part
        generated_sequences = []
        input_len = input_ids.shape[1]
        
        decoded = self.tokenizer.batch_decode(
            outputs[:, input_len:], 
            skip_special_tokens=True
        )
        
        # Reshape back to (B, G)
        result = []
        for i in range(len(prompts)):
            start = i * G
            end = start + G
            result.append(decoded[start:end])
            
        self.policy_model.train()
        
        # Restore padding side to right for training
        self.tokenizer.padding_side = "right"
        
        return result
        
    def _compute_rewards(self, prompts: List[str], generations: List[List[str]]) -> torch.Tensor:
        """
        Compute rewards for generations.
        
        Returns:
            Tensor of shape (B, G)
        """
        # Flatten
        flat_prompts = []
        flat_generations = []
        
        for prompt, gen_list in zip(prompts, generations):
            for gen in gen_list:
                flat_prompts.append(prompt)
                flat_generations.append(gen)
                
        # Call reward function
        # Returns tensor of shape (B * G,)
        rewards = self.reward_function(flat_prompts, flat_generations)
        
        # Reshape to (B, G)
        B = len(prompts)
        G = self.config.num_generations
        return rewards.view(B, G).to(self.policy_model.device)
        
    def _compute_loss(self, prompts: List[str], generations: List[List[str]], rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute GRPO loss.
        
        Args:
            prompts: List of B prompts
            generations: List of B lists of G generations
            rewards: Tensor of shape (B, G)
            
        Returns:
            Scalar loss
        """
        B = len(prompts)
        G = self.config.num_generations
        
        # Flatten for processing
        flat_prompts = []
        flat_texts = [] # prompt + generation
        
        for prompt, gen_list in zip(prompts, generations):
            for gen in gen_list:
                flat_prompts.append(prompt)
                flat_texts.append(prompt + gen)
                
        # Tokenize full texts
        inputs = self.tokenizer(
            flat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.policy_model.device)
        
        # Tokenize prompts only (to find lengths)
        prompt_inputs = self.tokenizer(
            flat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        )
        
        # Create labels (mask out prompt)
        labels = inputs.input_ids.clone()
        prompt_lens = prompt_inputs.attention_mask.sum(dim=1)
        
        for i, length in enumerate(prompt_lens):
            labels[i, :length] = -100 # Ignore prompt
            # Also ignore padding in inputs (already handled by CrossEntropy if -100, but let's be safe)
            labels[i, inputs.attention_mask[i] == 0] = -100
            
        # Forward pass Policy
        outputs = self.policy_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
        logits = outputs.logits # (B*G, seq_len, vocab)
        
        # Forward pass Reference
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            ref_logits = ref_outputs.logits
            
        # Compute log probs for the generated tokens
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Gather log probs of the actual tokens
        # shape: (B*G, seq_len-1)
        log_probs = -nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        ).view(shift_labels.size())
        
        ref_log_probs = -nn.functional.cross_entropy(
            shift_ref_logits.view(-1, shift_ref_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        ).view(shift_labels.size())
        
        # Mask out ignored tokens (-100)
        mask = (shift_labels != -100).float()
        log_probs = log_probs * mask
        ref_log_probs = ref_log_probs * mask
        
        # Sum log probs over the sequence (per generation)
        # shape: (B*G,)
        per_token_logps = log_probs.sum(dim=1)
        per_token_ref_logps = ref_log_probs.sum(dim=1)
        
        # Compute KL (approximate: log_p - log_ref_p)
        # This is actually the ratio log(p/ref)
        kl = per_token_logps - per_token_ref_logps
        
        # Compute Advantages
        # rewards shape: (B, G) -> flatten to (B*G,)
        flat_rewards = rewards.view(-1)
        
        # Normalize advantages within group
        # Reshape to (B, G) to compute stats
        rewards_grouped = flat_rewards.view(B, G)
        mean = rewards_grouped.mean(dim=1, keepdim=True)
        std = rewards_grouped.std(dim=1, keepdim=True)
        advantages = (rewards_grouped - mean) / (std + 1e-8)
        flat_advantages = advantages.view(-1)
        
        # GRPO Loss
        # Since we are doing 1 step, ratio is 1.0
        # Loss = - (Advantage - beta * KL)
        # We average over the batch
        
        # Note: In standard PPO, we maximize Objective. In PyTorch, we minimize Loss.
        # Objective = E[ A - beta * KL ]
        # Loss = - Objective = -A + beta * KL
        
        loss = -(flat_advantages - self.config.beta * kl)
        
        return loss.mean()

    def _collate_fn(self, batch):
        """Collate prompts."""
        return {"prompts": [item["prompt"] for item in batch]}
