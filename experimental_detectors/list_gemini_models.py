"""
List available Gemini models
"""

import google.generativeai as genai

# Hardcoded API key
api_key = "AIzaSyBqhaDLVDeBV2Et52WtP7IYu8lZOKF8vfg"

print("Listing available Gemini models...")

try:
    genai.configure(api_key=api_key)
    
    print("\nAvailable models:")
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  - {model.name}")
    
except Exception as e:
    print(f"Error: {e}")
