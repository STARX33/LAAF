# Phase 1: Episodic Memory Foundation
## Master Document Parser & Research Specialist

### Core Concept: Episodic Memory

**Episodic memory** is the autobiographical recall of specific past events, including the associated time, place, and emotions.

This is what we are building toward - a system that doesn't just store data, but creates "memory events" that can be recalled with full contextual richness.

---

## Phase 1 Implementation (Completed)

### What We Built

1. **Enhanced System Prompt with Personality Traits**
   - Scholar: Deep knowledge seeking and thorough analysis
   - Detective: Pattern recognition and hidden connection discovery
   - Analyst: Methodical information structuring
   - Synthesizer: Combining disparate information

2. **Web Research Tools** (`web_search_tool.py`)
   - `search_web_for_research()` - General web search with DuckDuckGo
   - `find_research_sources()` - Academic/technical source finder with relevance scoring

3. **Semantic Key Elements Extractor** (`document_tools.py`)
   - `extract_key_elements()` - Extracts top 10 semantic elements from documents
   - Uses n-gram analysis (unigrams, bigrams, trigrams)
   - Importance scoring and categorical classification
   - Confidence assessment

4. **Episodic Memory Storage** (`document_tools.py`)
   - `save_document_summary()` - Saves summaries with episodic metadata
   - Captures WHEN (timestamp, temporal markers)
   - Captures WHERE (location, spatial context)
   - Captures WHY (emotional significance, importance markers)
   - Stores recall triggers for future memory retrieval

5. **Folder Structure**
   - `rag/documents/` - Input documents for processing
   - `rag/summaries/` - Episodic memory events (JSON format)

---

## Episodic Memory Structure

Each processed document becomes a "memory event" with this structure:

```json
{
  "document_id": "unique_identifier",
  "memory_event": {
    "timestamp": "2025-11-29T10:30:00",  // WHEN
    "processed_date": "2025-11-29 10:30:00",
    "location": "rag/summaries"  // WHERE
  },
  "summary": {
    "executive_summary": "...",
    "key_elements": [...],
    "temporal_context": {...},  // Time references in content
    "spatial_context": {...},  // Place references in content
    "emotional_significance": {...},  // WHY it matters
    "research_sources": [...],
    "memory_triggers": [...]  // Keywords for recall
  },
  "episodic_metadata": {
    "memory_type": "document_analysis",
    "recall_triggers": [...],
    "emotional_markers": {...},
    "temporal_markers": {...},
    "spatial_markers": {...}
  }
}
```

---

## Workflow: From Document to Memory Event

1. **IDENTIFY** - Determine what documents need analysis
2. **EXTRACT** - Use tools (PDF, CSV, OCR) to get content
3. **ANALYZE** - Extract top 10 semantic key elements
4. **CONTEXTUALIZE** - Capture WHEN, WHERE, and emotional significance
5. **RESEARCH** - Find 3-5 web sources for validation
6. **CROSS-REFERENCE** - Align with RAG context
7. **SYNTHESIZE** - Create structured episodic memory
8. **SAVE** - Store with full metadata
9. **CONCLUDE** - Provide comprehensive analysis

---

## Evolution Path: Building Toward True Episodic Memory

### Phase 1: Foundation (Current) ✓
- Document parsing with semantic analysis
- Temporal, spatial, emotional metadata capture
- Web research integration
- Structured memory storage

### Phase 2: Semantic Clustering (Next)
- Group memories by topics
- Topic-based retrieval
- Folder-based organization

### Phase 3: Emotional Association
- Emotional trigger detection
- Feeling-based memory recall
- Sentiment analysis integration

### Phase 4: Adaptive Hierarchical Memory
- Importance-based retention
- Memory compression for efficiency
- Hierarchical RAG loading

### Phase 5: Similarity-Based Recall
- TF-IDF similarity search
- Find related memories
- Pattern-based retrieval

### Phase 6: Multi-Modal Episodic Memory
- Text + Image + Audio + Video memories
- Cross-modal associations
- Rich contextual recall

---

## Philosophy: Chris Sawyer Approach

**Minimalistic Excellence**
- Start simple (folder-based JSON)
- Evolve deliberately (no premature optimization)
- Privacy-first (local processing, no external APIs)
- Version-control friendly (human-readable JSON)

**Progressive Enhancement**
- Folder-based → Tensor-based → Vector database
- Build synthetic data first, then scale
- Each phase adds cognitive capability
- Never sacrifice clarity for cleverness

---

## Performance Targets (Phase 1)

- Document processing: <15s for 10-page PDF
- Key elements extraction: <5s
- Web search: <10s (3-5 sources)
- Total workflow: <30s end-to-end
- Success rate: 95%+
- Agent steps: 1-4 (no infinite loops)

---

## Technical Specifications

**Model**: qwen2:7b (7B parameters)
**Temperature**: 0.3 (deterministic tool calling)
**Max Steps**: 4 (prevents infinite loops)
**Architecture**: CodeAgent with local tools
**Storage**: Folder-based JSON (privacy-first)

---

## Dependencies

**Python Libraries**:
- smolagents (agentic framework)
- pytesseract, pdf2image, pypdf, pdfplumber (document processing)
- duckduckgo-search (web research)
- transformers, torch, Pillow (vision/ML)

**System Requirements**:
- Tesseract OCR engine
- Poppler (PDF to image conversion)
- Ollama with qwen2:7b model

---

## Usage Example

```python
# main.py configuration
MODEL_ID = "qwen2:7b"
TEMPERATURE = 0.3
MAX_STEPS = 4

# Phase 1 features enabled
ENABLE_WEB_RESEARCH = True
ENABLE_KEY_EXTRACTION = True
AUTO_SAVE_SUMMARIES = True

# Run agent
python main.py
```

The agent will:
1. Process documents from `rag/documents/`
2. Extract semantic key elements
3. Find relevant research sources
4. Create episodic memory structure
5. Save to `rag/summaries/` with full metadata

---

## Success Criteria

✓ Extract text from PDF successfully
✓ Generate top 10 key elements with >85% accuracy
✓ Web search returns 3-5 relevant sources
✓ Structured summary aligned with RAG context
✓ Save episodic memory JSON to rag/summaries/
✓ Complete in 1-4 steps (no loops)
✓ Maintain 95% success rate
✓ All v1.0 tests still pass

---

## What Makes This "Episodic"?

Traditional document parsing: **"Store text from a PDF"**

Episodic memory approach: **"Remember analyzing a technical document about synthetic neural enhancement on November 29th, 2025, which felt significant because it discussed cognitive architecture similar to our own system, and triggered associations with machine learning, adaptive systems, and privacy-first design"**

The difference:
- ✓ Temporal context (WHEN)
- ✓ Spatial context (WHERE)
- ✓ Emotional significance (WHY)
- ✓ Recall triggers (HOW to remember)
- ✓ Associated memories (WHAT else relates)

---

## Next Steps

1. **Test Phase 1** - Process sample documents
2. **Validate episodic structure** - Check JSON outputs
3. **Commit to git** - Save Phase 1 progress
4. **Begin Phase 2** - Implement memory clustering

---

**Repository**: https://github.com/STARX33/LAAF
**Version**: v1.0 → Phase 1 (Episodic Memory Foundation)
**Date**: November 29, 2025
