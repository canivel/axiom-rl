"""HuggingFace model wrapper for code generation."""

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import GeneratorConfig


class CodeGenerator:
    """
    HuggingFace-based code generator.

    Uses chat template for instruct models.
    """

    def __init__(self, config: GeneratorConfig):
        """
        Initialize the generator.

        Args:
            config: Generator configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        print(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        dtype = dtype_map.get(self.config.torch_dtype, "auto")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=self.config.device,
            trust_remote_code=True,
        )

        self._loaded = True
        device = next(self.model.parameters()).device
        print(f"Model loaded on device: {device}")

    def generate(
        self,
        messages: List[dict],
        num_samples: int = 1,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Generate responses for the given messages.

        Args:
            messages: Chat messages in OpenAI format [{"role": "...", "content": "..."}]
            num_samples: Number of responses to generate
            temperature: Override config temperature (optional)
            max_new_tokens: Override config max_new_tokens (optional)

        Returns:
            List of generated responses (decoded text only, without prompt)
        """
        if not self._loaded:
            self.load()

        temp = temperature if temperature is not None else self.config.temperature
        max_tokens = (
            max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        )

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_length = model_inputs.input_ids.shape[1]

        responses = []

        # Generate samples
        for _ in range(num_samples):
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the new tokens (exclude prompt)
            new_tokens = generated_ids[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)

        return responses

    def generate_batch(
        self,
        messages: List[dict],
        num_samples: int = 1,
    ) -> List[str]:
        """
        Generate multiple samples in a single forward pass (more efficient).

        Args:
            messages: Chat messages
            num_samples: Number of samples to generate

        Returns:
            List of generated responses
        """
        if not self._loaded:
            self.load()

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_length = model_inputs.input_ids.shape[1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                num_return_sequences=num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        responses = []
        for seq in generated_ids:
            new_tokens = seq[input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)

        return responses

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
