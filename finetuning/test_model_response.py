#!/usr/bin/env python3
"""
Test script for the trained conversational mental health model.
Tests the model's response to mental health prompts.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def load_model_and_tokenizer(base_model_path, adapter_path):
    """Load the base model and apply LoRA adapters."""
    
    print("🔧 Loading base model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("🔧 Loading LoRA adapters...")
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return model, tokenizer

def test_mental_health_response(model, tokenizer, user_input):
    """Test the model's response to a mental health prompt."""
    
    # Format the prompt like the training data
    prompt = f"""You are a compassionate mental health counselor. A client is seeking support. Please provide an empathetic, helpful response.

Client: {user_input}

Counselor:"""
    
    print(f"📝 Input prompt:\n{prompt}\n")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("🤖 Generating response...")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the counselor's response
    counselor_response = full_response.split("Counselor:")[-1].strip()
    
    return counselor_response

def main():
    """Main testing function."""
    
    print("🏥 Testing Conversational Mental Health Model")
    print("=" * 50)
    
    base_model_path = "../vanilla_deepseek/deepseek_r1_1.5b"
    
    # Allow model path as command line argument
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--model_path" and len(sys.argv) > 2:
        adapter_path = sys.argv[2]
    else:
        adapter_path = "./enhanced_counsel_model_20250805_170837"  # Use latest enhanced model
    
    # Test prompt
    test_prompt = "I've been feeling completely overwhelmed lately. No matter how much I try to keep up with everything, I always feel like I'm falling behind. It's starting to affect my sleep and my relationships. What should I do?"
    
    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer(base_model_path, adapter_path)
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model type: LoRA adapter on {base_model_path}")
        print(f"📊 Adapter size: ~141MB")
        print(f"📊 Total trainable parameters: 2.04% of base model")
        print()
        
        # Test response
        print("🧠 Testing mental health response...")
        print("=" * 50)
        
        response = test_mental_health_response(model, tokenizer, test_prompt)
        
        print(f"💬 Client: {test_prompt}")
        print()
        print(f"🏥 Counselor: {response}")
        print()
        print("=" * 50)
        
        # Analyze response quality
        print("📊 Response Analysis:")
        response_lower = response.lower()
        
        empathy_indicators = ["understand", "feel", "sounds", "difficult", "hard", "challenging"]
        support_indicators = ["help", "support", "together", "can", "try", "steps"]
        professional_indicators = ["strategies", "techniques", "coping", "manage", "approach"]
        
        empathy_score = sum(1 for word in empathy_indicators if word in response_lower)
        support_score = sum(1 for word in support_indicators if word in response_lower)
        professional_score = sum(1 for word in professional_indicators if word in response_lower)
        
        print(f"  - Empathy indicators: {empathy_score}/6")
        print(f"  - Support indicators: {support_score}/6") 
        print(f"  - Professional indicators: {professional_score}/5")
        print(f"  - Response length: {len(response.split())} words")
        
        if len(response) < 50:
            print("  ⚠️  Response seems quite short")
        elif len(response) > 400:
            print("  ⚠️  Response seems quite long")
        else:
            print("  ✅ Response length is appropriate")
            
        overall_score = empathy_score + support_score + professional_score
        print(f"  - Overall quality score: {overall_score}/17")
        
        if overall_score >= 8:
            print("  ✅ Good therapeutic response quality")
        elif overall_score >= 5:
            print("  ⚠️  Moderate therapeutic response quality")
        else:
            print("  ❌ Low therapeutic response quality")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the base model path exists and adapters are properly saved")

if __name__ == "__main__":
    main()