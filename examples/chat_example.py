#!/usr/bin/env python3
"""
Interactive chat example for Galena-2B model.

This script provides an interactive chat interface for the Galena-2B model,
allowing you to have a conversation focused on math and physics topics.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=" * 80)
    print("Galena-2B Interactive Chat")
    print("Specialized in Mathematics & Physics")
    print("=" * 80)
    print("\nLoading model...")

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

    print(f"Model loaded successfully on {model.device}!\n")
    print("Type 'exit', 'quit', or 'q' to end the conversation.")
    print("Type 'clear' to start a new conversation.\n")
    print("=" * 80 + "\n")

    # Conversation history
    messages = []

    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break

        # Check for clear command
        if user_input.lower() == 'clear':
            messages = []
            print("\nConversation cleared. Starting fresh!\n")
            continue

        # Skip empty input
        if not user_input:
            continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

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
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's latest response
        if "<|assistant|>" in full_response:
            assistant_response = full_response.split("<|assistant|>")[-1].strip()
        else:
            assistant_response = full_response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()

        # Add assistant response to history
        messages.append({"role": "assistant", "content": assistant_response})

        # Print response
        print(f"\nAssistant: {assistant_response}\n")

if __name__ == "__main__":
    main()
