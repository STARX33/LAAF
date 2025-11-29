# Model Upgrade Guide

This guide explains how to upgrade or switch models in LAAF for improved performance.

## Quick Model Switch

Edit `main.py` and change the `MODEL_ID` variable:

```python
# Model Configuration
MODEL_ID = "llama3"  # Change this line
```

## Recommended Models

### Current Generation (As of 2024)

| Model | Size | Best For | Notes |
|-------|------|----------|-------|
| `llama3` | 8B | General tasks | Good baseline, well-tested |
| `llama3.1` | 8B | General tasks | Improved over llama3 |
| `llama3.2` | 3B/8B | General tasks | Latest 3.x series |
| `qwen2.5` | 7B+ | General tasks | Strong reasoning |
| `qwen2.5-coder` | 7B+ | **Agentic tasks** | **RECOMMENDED for tool calling** |
| `mistral` | 7B | Fast inference | Efficient, good quality |

### Why Qwen2.5-Coder is Recommended for Agents

Qwen2.5-Coder is specifically trained for code generation and tool usage, making it excellent for:
- Writing Python code to call tools (what CodeAgent does)
- Understanding structured formats
- Following multi-step instructions
- Function calling patterns

## How to Upgrade

### Step 1: Pull the New Model

```bash
# Example: Upgrading to Qwen2.5-Coder
ollama pull qwen2.5-coder:7b
```

### Step 2: Update Configuration

Edit `main.py`:

```python
# Before
MODEL_ID = "llama3"

# After
MODEL_ID = "qwen2.5-coder:7b"
```

### Step 3: Test

```bash
python main.py
```

## Model-Specific Settings

### For Stronger Reasoning (Lower Temperature)

```python
MODEL_ID = "qwen2.5-coder:7b"
TEMPERATURE = 0.3  # More deterministic, better for tool calling
```

### For Creative Tasks (Higher Temperature)

```python
MODEL_ID = "llama3.2"
TEMPERATURE = 0.9  # More creative responses
```

### For Faster Inference (Smaller Models)

```python
MODEL_ID = "llama3.2:3b"  # 3B version, faster but less capable
TEMPERATURE = 0.7
```

## Advanced: Custom Model Parameters

You can also pass additional Ollama parameters in `ollama_model.py`:

```python
# In ollama_model.py, modify the payload in generate():
payload = {
    "model": self.model_id,
    "messages": ollama_messages,
    "stream": False,
    "options": {
        "temperature": kwargs.get("temperature", self.temperature),
        "top_p": 0.9,           # Nucleus sampling
        "top_k": 40,            # Top-k sampling
        "num_predict": 512,     # Max tokens to generate
        "repeat_penalty": 1.1,  # Reduce repetition
    }
}
```

## Troubleshooting

### Model Not Found

```bash
Error: model 'qwen2.5-coder' not found
```

**Solution**: Pull the model first
```bash
ollama pull qwen2.5-coder
```

### Out of Memory

If you get OOM errors with larger models:

1. Use smaller variants:
   ```python
   MODEL_ID = "llama3.2:3b"  # Instead of 8b
   ```

2. Or use quantized versions:
   ```bash
   ollama pull qwen2.5-coder:7b-q4_0  # 4-bit quantization
   ```

### Model Takes Too Long

Reduce the context or use smaller models:
```python
MODEL_ID = "llama3.2:3b"
MAX_STEPS = 4  # Reduce max steps
```

## Future Models

As new models are released on Ollama, you can use them immediately:

```bash
# Check available models
ollama list

# Pull any new model
ollama pull <model-name>

# Update MODEL_ID in main.py
```

## Performance Comparison

Based on community feedback (update as you test):

| Model | Speed | Quality | Tool Calling | Memory |
|-------|-------|---------|--------------|--------|
| llama3.2:3b | ⚡⚡⚡ | ⭐⭐ | ⭐⭐ | 2GB |
| llama3:8b | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | 4GB |
| qwen2.5-coder:7b | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 4GB |
| llama3.1:8b | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 4GB |

## Recommended Setup for Production

```python
# main.py
MODEL_ID = "qwen2.5-coder:7b"  # Best for agentic tasks
TEMPERATURE = 0.3              # Focused, deterministic
MAX_STEPS = 6                  # Reasonable limit
```

This configuration provides the best balance of:
- ✅ Reliable tool calling
- ✅ Consistent results
- ✅ Good reasoning quality
- ✅ Reasonable speed

## Testing New Models

When testing a new model, use the test script:

```bash
# Edit test_improved_implementation.py to use new model
python test_improved_implementation.py

# If tests pass, update main.py
python main.py
```

## Community Recommendations

Share your findings! If you discover a model that works particularly well for agentic RAG tasks, please contribute to the project documentation.

Good models for agentic workflows typically have:
- Strong instruction following
- Good structured output generation
- Reliable tool/function calling
- Consistent behavior (not too creative)

---

**Last Updated**: 2025-01-XX
**LAAF Version**: 1.0
**Ollama Compatibility**: All versions
