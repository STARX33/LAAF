
# 🧠 Local Agentic AI Framework (LAAF)

A modular, privacy first agentic AI framework designed to run fully **offline**, enabling users to build adaptable, context aware assistants using **a customized agent loop inspired by SmolAgents**, **local LLaMA 3 (via Ollama)**, and plug and play tools for vision, OCR, TTS, and RAG.

---

## 🚀 Project Vision

LAAF is not here to replace UI/UX — it’s the first step toward building a new paradigm: **Agentic Experience (AX)**.
AX is about working *with* intelligent agents who can see what you see, read what you read, and assist with daily or domain specific tasks — not just chat. Think of it as the groundwork for agents that can help fill out forms, interpret dashboards, summarize reports, or provide accessibility support — all **running privately, locally, and modularly.**
LAAF enables this future by combining local LLaMA 3, lightweight toolchains, and modular context injection (RAG) — giving developers full control to prototype and build agentic workflows that **go beyond conversation**.

It’s designed for:
- Developers building **domain-specific AI assistants**
- Accessibility advocates supporting **visually impaired users**
- Engineers experimenting with **image + text-based RAG**
- Anyone wanting to **run agentic AI fully locally**
- HIPPA compliant applications in **legal, medical, and business domains**

---

## 🧩 Key Features

- 🤖 **Local LLaMA 3 via Ollama**  
  Seamlessly integrated with our custom agent loop for smart, prompt aware behavior.

- 🖼️ **Screenshot-Based Visual Input**  
  Instead of heavy video processing, users can request help and trigger on the  spot screenshots.

- 🧾 **Optional OCR & TTS Modules (Offline)**  
  For reading and voicing visual/text content—great for accessibility. Powered by local engines such as **Tesseract** (OCR) and **pyttsx3** or **Coqui TTS** (TTS).

- 🗂️ **RAG Folders for Context Injection**  
  Users can drop documents, images, and more into structured folders to inject real-time knowledge into the AI’s reasoning.

- 🔐 **Privacy-First & Offline by Default**  
  No cloud tokens, no tracking, and no dependencies on online APIs.

---

## 📚 Use Case Scenarios

This system is built to adapt to real-world needs across industries:

### ⚖️ Legal / Law Firms
- Upload contracts, litigation records, or case notes into `rag/text/contracts/`.
- Ask the AI to summarize, compare clauses, or identify precedents.
- Use OCR for scanned legal documents or handwritten notes.

### 🏥 Medical Clinics
- Reference diagnostic images or de-identified patient records in `rag/images/`.
- Use OCR to parse charts or intake forms.
- Add TTS for auditory summaries in accessibility mode.

### 🏢 Business Operations / HR
- Load company manuals, compliance docs, or SOPs.
- Ask questions like "What’s our vacation policy?" or "Show me the onboarding steps."
- Visual context from screenshots of CRM or HR dashboards improves response accuracy.

### 🧰 Field Technicians / Hardware Support
- Upload labeled equipment photos and repair manuals to `rag/images/tools/`.
- Snap a picture of malfunctioning hardware — get diagnostic help instantly.
- Great for service techs, remote support, or DIY systems.

### 🎓 Education & Accessibility
- Teachers can preload lesson materials or scanned worksheets.
- Students can request help and get auditory feedback via TTS.
- OCR helps interpret handwritten or printed worksheets on the fly.

---

## 📁 Directory Structure

```
LAAF/
│
├── main.py                  # Core runner script
├── ollama_model.py          # LLaMA 3 model wrapper
├── rag_loader.py            # Loads RAG context
├── tools.py                 # Toolchain (e.g. suggest_menu)
├── vision_tools.py          # Vision tools: BLIP, OWL-ViT
├── requirements.txt
├── README.md
├── rag/
│   ├── images/              # Reference image folders
│   │   └── Folder_1/
│   └── text/                # Text RAG memory
│       └── Folder_1/
```

---

## 🛠️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
> Requires Python 3.10+ and Ollama running locally with `llama3`

### 2. Start Ollama
```bash
ollama run llama3
```

### 3. Launch the Agent
```bash
python main.py
```

---

## 🔐 Environment Setup

LAAF is designed to run fully offline. No external tokens or APIs are required by default.

If you'd like to configure optional features, create a `.env` file with flags:
```env
USE_OCR=true
USE_TTS=true
RAG_MODE=full
```
These can also be passed through CLI arguments or config settings depending on your implementation.

---

## 📚 RAG (Retrieval Augmented Generation)

Structure your own data references:

- `rag/images/<category>` — drop example images (e.g. tools, diagrams, forms)
- `rag/text/<topic>` — add `.txt`, `.csv`, or `.md` documents (e.g. policies, transcripts, reports)

The agent will search these folders automatically when relevant to a query.

If folders are empty, the system still runs — just with less context.

---

## ♿ Accessibility Mode (Optional)

Enable via config or `.env` flags:
```yaml
use_ocr: true
use_tts: true
```

---

## 🔬 Testing Scenarios

- ✅ **No RAG:** Should fall back to image only reasoning
- ✅ **Text RAG only:** Inject `.txt` memory like "This is a robot disguised as a dog."
- ✅ **Image RAG only:** Drop `robot_dog.jpg` — expect image captioning to enhance inference
- ✅ **Full RAG:** Combine image + text context for deepest analysis

**Note:** It is normal for some local runs to hang. If that happens, stop and re-run once or twice — LLaMA via Ollama can sometimes loop unpredictably on the first inference.

---

## 🙏 Acknowledgments

This project was originally inspired by the [Hugging Face Agents course](https://huggingface.co/learn/agents) — especially the "Alfred" example.

LAAF extends that foundation with:
- Local-first LLaMA 3 via Ollama
- Image + Text RAG system
- BLIP / OWL-ViT vision tools
- Offline-first, modular architecture

We have departed from the original `smolagents` library to build our own modular agent loop that operates fully offline, locally, and securely without needing Hugging Face tokens or hosted APIs.

Massive thanks to the Hugging Face team for open sourcing the core ideas.

---

## 👥 Contributing

Pull requests and feature ideas welcome!  
Submit through GitHub Issues or PRs.

---

## 🔒 License

MIT License — free and open source with attribution.

---

## 🧭 Final Word

Agentic software doesn’t have to live in the cloud.  
With **LAAF**, you're free to build intelligent assistants that run where you are — with your data, your vision, and your control.

Let’s build the agentic future — together.
