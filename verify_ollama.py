#!/usr/bin/env python3
"""
NEXUS AI v4.0 — Ollama Local Setup Verification
Checks Docker, models, and connection before startup
"""

import subprocess
import requests
import time
import sys
from pathlib import Path

def check_docker():
    """Verify Docker daemon is running"""
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Docker not available: {e}")
        return False

def check_ollama_container():
    """Check if Ollama container is running"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=ollama', '--format', '{{.Names}}'],
            capture_output=True, text=True, timeout=5
        )
        return 'ollama' in result.stdout
    except Exception as e:
        print(f"❌ Cannot check container: {e}")
        return False

def check_ollama_api():
    """Verify Ollama API is responding"""
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"  Waiting for Ollama API... ({retry_count + 1}/{max_retries})", end='\r')
        time.sleep(1)
        retry_count += 1
    
    return False

def get_available_models():
    """List downloaded models"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            return [m['name'].split(':')[0] for m in models]
    except:
        pass
    return []

def main():
    print("=" * 60)
    print("NEXUS AI v4.0 — Ollama Setup Verification")
    print("=" * 60)
    print()
    
    # Check Docker
    print("📦 Checking Docker...")
    if not check_docker():
        print("❌ Docker daemon not running. Start Docker Desktop and try again.")
        return False
    print("✅ Docker is running")
    print()
    
    # Check Ollama container
    print("🐳 Checking Ollama container...")
    if not check_ollama_container():
        print("❌ Ollama container not running")
        print("   Starting container: docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama:latest")
        return False
    print("✅ Ollama container is running")
    print()
    
    # Check Ollama API
    print("🔌 Checking Ollama API (http://localhost:11434)...")
    if not check_ollama_api():
        print("❌ Ollama API not responding after 30 seconds")
        print("   Try: docker logs ollama")
        return False
    print("✅ Ollama API is responsive")
    print()
    
    # List models
    print("📚 Available models:")
    models = get_available_models()
    if models:
        for model in models:
            print(f"   ✓ {model}")
    else:
        print("   ⚠️  No models downloaded yet")
    print()
    
    print("=" * 60)
    print("✅ Setup verification complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Download models if needed:")
    print("   docker exec ollama ollama pull mistral")
    print("   docker exec ollama ollama pull neural-chat")
    print()
    print("2. Configure .env file with OLLAMA_HOST=localhost:11434")
    print()
    print("3. Start NEXUS AI:")
    print("   python main.py")
    print()
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
