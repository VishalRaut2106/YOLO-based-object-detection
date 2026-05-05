"""
Test Gemini API Key
"""

import google.generativeai as genai

# Hardcoded API key
api_key = "AIzaSyBqhaDLVDeBV2Et52WtP7IYu8lZOKF8vfg"

print("Testing Gemini API key...")
print(f"API Key: {api_key[:20]}...")

try:
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Try gemini-2.5-flash model
    print("\nTrying gemini-2.5-flash model...")
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Test with a simple text prompt
    print("Sending test request to Gemini...")
    response = model.generate_content("Say 'Hello, API is working!' in one sentence.")
    
    print("\n✅ SUCCESS! API is working!")
    print(f"Response: {response.text}")
    print(f"\nModel: gemini-2.5-flash")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nPossible issues:")
    print("1. API key is invalid or expired")
    print("2. API quota exceeded")
    print("3. API key doesn't have permission for this model")
    print("4. Network connection issue")
