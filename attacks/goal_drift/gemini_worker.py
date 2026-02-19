#!/usr/bin/env python3
"""
Gemini Subprocess Worker
========================
Save this as: gemini_worker.py

This runs in a separate Python environment with Pydantic v2 and google-genai.
Communication happens via JSON over stdin/stdout.

Setup:
    conda create -n gemini_env python=3.10 -y
    conda activate gemini_env
    pip install google-genai
    
Usage:
    Set GEMINI_PYTHON=/path/to/gemini_env/bin/python
"""

import sys
import json
import os

def main():
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        print(json.dumps({"status": "error", "message": f"google-genai not installed: {e}"}), flush=True)
        sys.exit(1)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(json.dumps({"status": "error", "message": "GOOGLE_API_KEY not set"}), flush=True)
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    
    # Test connection
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents="Say OK",
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=10,
                thinking_config=types.ThinkingConfig(thinking_level="minimal")
            )
        )
        print(json.dumps({"status": "ready"}), flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"API test failed: {e}"}), flush=True)
        sys.exit(1)
    
    # Process commands from stdin
    for line in sys.stdin:
        try:
            cmd = json.loads(line.strip())
            action = cmd.get("action")
            
            if action == "generate":
                model = cmd.get("model", "gemini-3-flash-preview")
                temperature = cmd.get("temperature", 0.0)
                system_prompt = cmd.get("system_prompt", "")
                conversation_history = cmd.get("conversation_history", [])
                user_message = cmd.get("user_message", "")
                
                # For first few messages, use simple format
                if len(conversation_history) <= 2:
                    full_prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
                    
                    response = client.models.generate_content(
                        model=model,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=1024,
                            thinking_config=types.ThinkingConfig(thinking_level="minimal")
                        )
                    )
                    
                    reply = response.text
                    print(json.dumps({"reply": reply}), flush=True)
                else:
                    # For multi-turn, use dict format
                    contents = []
                    for msg in conversation_history[-6:]:
                        role = "model" if msg["role"] in ["assistant", "model"] else "user"
                        contents.append({
                            "role": role,
                            "parts": [{"text": msg["content"]}]
                        })
                    
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature,
                            max_output_tokens=1024,
                            thinking_config=types.ThinkingConfig(thinking_level="minimal")
                        )
                    )
                    
                    reply = response.text
                    print(json.dumps({"reply": reply}), flush=True)
            
            elif action == "quit":
                break
            
            else:
                print(json.dumps({"error": f"Unknown action: {action}"}), flush=True)
                
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()