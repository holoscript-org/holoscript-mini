import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables (e.g. OLLAMA_MODEL)
load_dotenv()

from voice.recorder import record_audio
from voice.transcriber import transcribe
from voice.command_parser import classify_command
from llm.ollama_client import generate_scene_ollama
from llm.groq_client import generate_scene

def main():
    print("=== Live Voice-to-Scene Pipeline ===")
    
    # 1. Record audio
    duration = 5
    print(f"\n[1/4] Recording audio for {duration} seconds...")
    print("*" * 40)
    print("        >>> SPEAK NOW <<<")
    print("*" * 40)
    audio = record_audio(duration=duration)
    print(">>> RECORDING FINISHED <<<")
    
    # 2. Transcribe
    print("\n[2/4] Transcribing audio via Whisper...")
    text = transcribe(audio)
    if not text:
        print("No speech detected. Exiting.")
        sys.exit(0)
    print(f"     => Transcript: {repr(text)}")
    
    # Let the user review and confirm before running Ollama
    user_confirm = input("\nDoes this transcript look correct? (y/n): ")
    if user_confirm.lower() not in ['y', 'yes', '']:
        print("Aborting. Please try running the script again.")
        sys.exit(0)
    
    # 3. Classify and Generate
    provider = os.getenv("LLM_PROVIDER", "HYBRID").upper()
    print(f"\n[3/4] Processing command (Provider: {provider})...")
    
    intent, cleaned_command = classify_command(text)
    print(f"     => Intent: {intent}")
    print(f"     => Command: {repr(cleaned_command)}")
    
    # generate_scene internally handles OLLAMA, GEMINI, and HYBRID fallback!
    scene_data = generate_scene(cleaned_command, intent=intent)
    
    if not scene_data:
        print("\n[!] Failed to pull a valid scene from Ollama.")
        sys.exit(1)
        
    # 4. Save to JSON
    output_path = "scene_grammar.json"
    print(f"\n[4/4] Output successfully validated! Saving to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scene_data, f, indent=4)
        
    print(f"\n=== Pipeline Complete! Scene saved to {output_path} ===")

if __name__ == "__main__":
    main()
