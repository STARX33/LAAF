
# 🧠 Local Agentic AI Framework (LAAF)

A modular, privacy-first agentic AI framework designed to run fully **offline**, enabling users to build adaptable, context-aware assistants using **smolagents**, **local Ollama models**, and plug-and-play tools for vision, OCR, TTS, and RAG.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## 🚀 Project Vision

LAAF is not here to replace UI/UX — it's the first step toward building a new paradigm: **Agentic Experience (AX)**.
AX is about working *with* intelligent agents who can see what you see, read what you read, and assist with daily or domain-specific tasks — not just chat. Think of it as the groundwork for agents that can help fill out forms, interpret dashboards, summarize reports, or provide accessibility support — all **running privately, locally, and modularly.**

LAAF enables this future by combining local LLMs via Ollama, lightweight toolchains, and modular context injection (RAG) — giving developers full control to prototype and build agentic workflows that **go beyond conversation**.

It's designed for:
- Developers building **domain-specific AI assistants**
- Accessibility advocates supporting **visually impaired users**
- Engineers experimenting with **image + text-based RAG**
- Anyone wanting to **run agentic AI fully locally**
- HIPAA-compliant applications in **legal, medical, and business domains**

---

## ⚡ Quick Start

### Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed and running ([Download here](https://ollama.ai/))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Alfred.git
cd Alfred

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull an Ollama model (recommended: qwen2:7b)
ollama pull qwen2:7b

# 5. Run the agent!
python main.py
```

### First Run

The agent will:
1. Load RAG context from `rag/text/` and `rag/images/`
2. Analyze the latest image in `input_images/`
3. Provide a conclusion in **1-4 steps** (typically completes in ~9 seconds)

---

## 🧩 Key Features

### 🤖 Local LLM via Ollama
- Seamlessly integrated with smolagents CodeAgent framework
- Easy model switching (see [MODEL_UPGRADE_GUIDE.md](MODEL_UPGRADE_GUIDE.md))
- Optimized for tool calling and agentic workflows

### 🖼️ Vision Tools
- **BLIP** image captioning
- Screenshot-based visual input
- Multi-image analysis support

### 🗂️ RAG (Retrieval-Augmented Generation)
- Text context injection from `rag/text/`
- Image context from `rag/images/` (automatically captioned)
- Zero-configuration - just drop files in folders

### 🔐 Privacy-First & Offline
- No cloud tokens, no tracking, no online APIs
- 100% local inference
- HIPAA-compliant ready

### 🛡️ Production-Ready
- ✅ No infinite loops (max_steps protection)
- ✅ Proper error handling
- ✅ Token usage tracking
- ✅ Comprehensive logging

---

## 📁 Directory Structure

```
LAAF/
│
├── main.py                  # Core runner script (START HERE)
├── ollama_model.py          # Ollama model wrapper (smolagents-compatible)
├── rag_loader.py            # Loads RAG context from folders
├── tools.py                 # Core tools (final_answer, etc.)
├── vision_tools.py          # Vision tools (BLIP, image processing)
├── requirements.txt         # Python dependencies
│
├── rag/
│   ├── images/              # Reference images for context
│   │   └── Folder_1/        # Organize by category
│   └── text/                # Text documents for context
│       └── Folder_1/        # Organize by topic
│
├── input_images/            # Images to analyze (agent reads latest)
│
├── results/                 # Agent output logs
│
└── docs/
    ├── README.md            # This file
    ├── MODEL_UPGRADE_GUIDE.md
    └── IMPLEMENTATION_SUMMARY.md
```

---

## 🛠️ Configuration

All configuration is in **`main.py`** (lines 22-29):

```python
# Model Configuration
MODEL_ID = "qwen2:7b"       # Change model here (see MODEL_UPGRADE_GUIDE.md)
TEMPERATURE = 0.3           # Lower = focused, Higher = creative
MAX_STEPS = 4               # Prevent infinite loops

# RAG Configuration
ENABLE_TEXT_RAG = True      # Load text documents
ENABLE_IMAGE_RAG = True     # Load image captions
```

### Recommended Settings

**For Production Use:**
```python
MODEL_ID = "qwen2:7b"       # Best balance (or qwen2.5-coder:7b)
TEMPERATURE = 0.3           # Deterministic, fast
MAX_STEPS = 4               # Efficient completion
```

**For Experimentation:**
```python
MODEL_ID = "llama3.2"
TEMPERATURE = 0.7           # More creative
MAX_STEPS = 6               # Allow more reasoning
```

---

## 📚 Use Case Scenarios

### ⚖️ Legal / Law Firms
- Upload contracts, litigation records, case notes to `rag/text/contracts/`
- Ask the AI to summarize, compare clauses, or identify precedents
- Use OCR for scanned legal documents

### 🏥 Medical Clinics
- Reference diagnostic images in `rag/images/`
- Use OCR to parse charts or intake forms
- Add TTS for auditory summaries in accessibility mode

### 🏢 Business Operations / HR
- Load company manuals, compliance docs, SOPs
- Ask questions like "What's our vacation policy?"
- Visual context from screenshots improves accuracy

### 🧰 Field Technicians / Hardware Support
- Upload labeled equipment photos and repair manuals to `rag/images/tools/`
- Snap pictures of malfunctioning hardware — get diagnostic help instantly

### 🎓 Education & Accessibility
- Preload lesson materials or scanned worksheets
- Students get auditory feedback via TTS
- OCR interprets handwritten or printed worksheets

---

## 🔬 Performance Benchmarks

Tested on the "robot dog" scenario (image analysis + RAG context alignment):

| Model | Steps | Time | Quality | Tool Calling |
|-------|-------|------|---------|--------------|
| llama3 (TEMP=0.7) | 6 | 37s | Good | Poor (repeated calls) |
| qwen2:7b (TEMP=0.7) | 6 | 24s | Good | Good |
| **qwen2:7b (TEMP=0.3)** | **1** | **9s** | **Excellent** | **Perfect** ✅ |

**Key Findings:**
- Lower temperature (0.3) dramatically improves efficiency
- Qwen2 models excel at tool calling vs Llama3
- Proper configuration = 6x speedup

---

## 🔄 Model Upgrades

LAAF is designed for easy model switching. See **[MODEL_UPGRADE_GUIDE.md](MODEL_UPGRADE_GUIDE.md)** for:
- How to upgrade to larger/better models
- Recommended models for different use cases
- Performance comparisons
- Troubleshooting

**Quick upgrade:**
```bash
ollama pull qwen2.5-coder:7b
```

Then edit `main.py` line 23:
```python
MODEL_ID = "qwen2.5-coder:7b"
```

---

## 🧪 Testing

### Quick Test

```bash
python test_improved_implementation.py
```

Expected output:
```
1. Testing Model Interface Compatibility... [OK]
2. Testing Message Format Compatibility... [OK]
3. Testing CodeAgent Integration... [OK]
4. Testing Simple Agent Execution... [OK]

ALL TESTS PASSED!
```

### Full Integration Test

```bash
python main.py
```

Should complete in 1-4 steps with proper final answer.

---

## 🔐 Environment Setup

LAAF runs fully offline by default. No external tokens or APIs are required.

Optional `.env` configuration:
```env
# Optional features
USE_OCR=true
USE_TTS=true
RAG_MODE=full
```

---

## 🤝 Contributing

Pull requests and feature ideas welcome!

**Areas for contribution:**
- Additional tool implementations
- New model adapters
- Performance optimizations
- Documentation improvements

Submit through GitHub Issues or PRs.

---

## 📝 License

MIT License — free and open source with attribution.

---

## 🙏 Acknowledgments

This project was originally inspired by the [Hugging Face Agents course](https://huggingface.co/learn/agents) — especially the "Alfred" example.

LAAF extends that foundation with:
- Local-first LLM inference via Ollama
- Production-ready smolagents integration
- Image + Text RAG system
- BLIP / OWL-ViT vision tools
- Offline-first, modular architecture

Massive thanks to:
- **Hugging Face** for smolagents and the foundational agent concepts
- **Ollama** for making local LLM inference accessible
- **Salesforce** for BLIP image captioning models

---

## 🧭 Final Word

Agentic software doesn't have to live in the cloud.
With **LAAF**, you're free to build intelligent assistants that run where you are — with your data, your vision, and your control.

Let's build the agentic future — together.

---

## 📚 Additional Documentation

- **[MODEL_UPGRADE_GUIDE.md](MODEL_UPGRADE_GUIDE.md)** - How to upgrade models
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[SECURITY.md](SECURITY.md)** - Security considerations

---

## 🐛 Troubleshooting

### Agent loops infinitely
- Check `MAX_STEPS` is set (recommended: 4)
- Try lowering `TEMPERATURE` to 0.3
- Ensure you're using a Qwen model (better at tool calling)

### "Ollama API request failed"
- Ensure Ollama is running: `ollama serve`
- Check model is installed: `ollama list`
- Pull model if needed: `ollama pull qwen2:7b`

### "No image found"
- Ensure an image exists in `input_images/`
- Supported formats: .png, .jpg, .jpeg

### Import errors (PIL, transformers, etc.)
```bash
pip install -r requirements.txt --upgrade
```

---

**Version:** 1.0
**Last Updated:** 2025-01-28
**Tested with:** Python 3.10+, Ollama 0.1.0+, smolagents 1.17.0
