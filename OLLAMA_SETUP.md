# NEXUS AI v4.0 — Ollama Local Setup Complete ✅

**Date:** 2026-07-06  
**System:** Windows 11 Pro | Intel i7-12800H | 64GB RAM | Docker 28.3.0

---

## 🎉 Setup Summary

### What's Been Installed

✅ **Ollama** — Running in Docker container (port 11434)  
✅ **Mistral 7B** — Primary LLM (4.4GB) - Fast & accurate  
✅ **Neural-Chat 7B** — Secondary model (4.1GB) - Great for conversations  
✅ **Python Client** — `ollama` package installed (v0.6.2)  
✅ **Verification Script** — `verify_ollama.py` for health checks  
✅ **Startup Script** — `start-nexus.bat` for one-click launch  

---

## 🚀 Quick Start

### Option 1: Interactive Launcher (Easiest)
```batch
start-nexus.bat
```
This starts Ollama, verifies connectivity, and launches NEXUS AI.

### Option 2: Manual Steps
```powershell
# Verify setup
python verify_ollama.py

# Start NEXUS AI
python main.py
```

### Option 3: With Custom Model
```powershell
# Download additional model
docker exec ollama ollama pull llama2

# Run with specific model (via .env.ollama)
python main.py
```

---

## 📊 Ollama Models Available

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| **mistral** | 4.4GB | ⚡ Fast | General tasks, routing |
| **neural-chat** | 4.1GB | ⚡ Fast | Conversations, dialogue |

### Optional Models to Add

```powershell
# High-performance 13B model
docker exec ollama ollama pull llama2-13b

# Specialized coding
docker exec ollama ollama pull codellama

# Ultra-fast (but less accurate)
docker exec ollama ollama pull neural-chat-tiny
```

---

## ⚙️ Configuration Files

### `.env.ollama` — Ollama Backend Settings
- `OLLAMA_HOST=localhost:11434` — API endpoint
- `LLM_MODEL_PRIMARY=mistral` — Default model
- `OLLAMA_NUM_PARALLEL=4` — Concurrent requests
- Tuned for your i7-12800H hardware

### Docker Container
```bash
# View logs
docker logs ollama -f

# Stop container
docker stop ollama

# Resume container
docker start ollama

# Remove container (keep models)
docker rm ollama

# View storage
docker volume inspect ollama
```

---

## 🔧 Customization

### Switch Primary Model
Edit `.env.ollama`:
```env
OLLAMA_MODEL_PRIMARY=neural-chat  # or llama2, codellama, etc.
```

### Increase Model Parallelism
For better throughput with your 14-core CPU:
```env
OLLAMA_NUM_PARALLEL=8  # More concurrent requests
OLLAMA_NUM_THREADS=12  # More CPU threads
```

### Enable GPU Acceleration (if available)
```env
CUDA_VISIBLE_DEVICES=0  # Use discrete GPU
```

---

## 🆘 Troubleshooting

### Ollama API not responding
```powershell
# Restart container
docker restart ollama

# Wait 10 seconds
Start-Sleep -Seconds 10

# Test connection
python verify_ollama.py
```

### Models downloading slowly
- Download happens in background — you can start using while downloading
- Mistral: 4.4GB (~5-10 min on typical internet)
- Neural-Chat: 4.1GB (~5-10 min on typical internet)

### Out of memory (unlikely with 64GB)
- Reduce parallelism: `OLLAMA_NUM_PARALLEL=1`
- Or download smaller models: `neural-chat-tiny`

### Docker daemon not running
- Start Docker Desktop application
- Or: `wsl --shutdown && docker run hello-world`

---

## 📈 Performance Expectations

### Latency (First Response)
- **Mistral**: 2-5 seconds
- **Neural-Chat**: 1-3 seconds

### Throughput
- **Tokens/sec**: 30-50 tokens/second (depends on model & hardware)
- **Concurrent Requests**: Up to 4 parallel

### Memory Usage
- **Mistral loaded**: ~8-10GB
- **Both models**: ~15-20GB (total 64GB available)

---

## 📚 Next Steps

1. **Launch Application**
   ```batch
   start-nexus.bat
   ```

2. **Test LLM Routing**
   - Check `nexus_brain/llm_router.py` configuration
   - Simple queries → Mistral
   - Complex queries → Can use Groq fallback (if configured)

3. **Monitor Performance**
   ```bash
   docker stats ollama
   ```

4. **Add More Models (Optional)**
   ```bash
   docker exec ollama ollama pull llama2
   docker exec ollama ollama pull codellama
   ```

---

## 🔗 Useful Links

- **Ollama Models**: https://ollama.ai/library
- **Docker Documentation**: https://docs.docker.com
- **NEXUS AI Docs**: See `README.md`
- **Ollama API**: `http://localhost:11434/api/docs`

---

## ✨ What You Can Do Now

✅ Run NEXUS AI completely offline (no API costs!)  
✅ Process private conversations locally  
✅ Use multiple models via intelligent routing  
✅ Scale models based on task complexity  
✅ Leverage your powerful i7-12800H hardware  
✅ Fallback to Groq API for heavy workloads (optional)  

---

**Status:** ✅ **Ready to Launch**

Run `start-nexus.bat` or `python main.py` to begin!
