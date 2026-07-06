#!/usr/bin/env python3
"""Quick test of Ollama inference"""

from ollama import Client

client = Client(host='localhost:11434')
print("Testing Ollama Mistral model...")
print()

response = client.generate(
    model='mistral',
    prompt='What is 2+2? Give a brief answer.',
    stream=False
)

print("✅ Ollama Inference Working!")
print()
print("Response from Mistral:")
print("-" * 50)
print(response['response'])
print("-" * 50)
