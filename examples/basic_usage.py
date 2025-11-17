#!/usr/bin/env python3
"""
Basic usage example for Galena-2B model.

This script demonstrates how to load and use the Galena-2B model
for simple inference tasks.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("Loading Galena-2B model...")

    # Path to the model (adjust if needed)
    model_path = "models/math-physics/hf"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print(f"Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")

    # Example prompts
    prompts = [
        "Calculate the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
        "Explain Newton's second law of motion.",
        "What is the escape velocity from Earth's surface?",
    ]

    print("\n" + "=" * 80)
    print("Running inference on example prompts...")
    print("=" * 80 + "\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"Example {i}:")
        print(f"Prompt: {prompt}\n")

        # Format as chat message
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1
            )

        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        print(f"Response: {response}\n")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    main()
