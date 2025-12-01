"""
Proper Ollama model wrapper for smolagents framework.
Implements the smolagents.models.Model interface for local Ollama models.
"""
import requests
from typing import Optional
from smolagents.models import Model, ChatMessage
from smolagents.monitoring import TokenUsage


class OllamaModel(Model):
    """
    Ollama model wrapper compatible with smolagents CodeAgent.

    Supports easy model switching for upgrades:
    - llama3, llama3.1, llama3.2, llama3.3
    - qwen2.5, qwen2.5-coder (excellent for agentic tasks)
    - mistral, mixtral
    - And any other Ollama-supported model

    Args:
        model_id: Ollama model identifier (default: "llama3")
        base_url: Ollama API base URL (default: "http://localhost:11434")
        temperature: Sampling temperature (default: 0.7)
        **kwargs: Additional arguments passed to Model base class
    """

    def __init__(
        self,
        model_id: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        **kwargs
    ):
        # Initialize parent Model class
        super().__init__(model_id=model_id, **kwargs)
        self.base_url = base_url
        self.temperature = temperature

    def generate(
        self,
        messages: list[dict[str, str] | ChatMessage],
        stop_sequences: Optional[list[str]] = None,
        response_format: Optional[dict[str, str]] = None,
        tools_to_call_from: Optional[list] = None,
        **kwargs
    ) -> ChatMessage:
        """
        Generate a response using Ollama's chat API.

        Args:
            messages: List of message dicts or ChatMessage objects
            stop_sequences: Optional stop sequences
            response_format: Optional response format specification
            tools_to_call_from: Optional list of tools (unused by Ollama but required by interface)
            **kwargs: Additional generation parameters

        Returns:
            ChatMessage with the model's response
        """
        # Convert messages to Ollama format
        ollama_messages = self._convert_messages(messages)

        # Prepare request payload
        payload = {
            "model": self.model_id,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
            }
        }

        # Add stop sequences if provided
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        # Make request to Ollama chat API
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120  # 2 minute timeout for long generations
            )
            response.raise_for_status()
            result = response.json()

            # Extract response content
            content = result.get("message", {}).get("content", "").strip()

            # Extract token usage if available
            prompt_tokens = result.get("prompt_eval_count", 0)
            completion_tokens = result.get("eval_count", 0)

            # Return ChatMessage with proper token usage
            return ChatMessage(
                role="assistant",
                content=content,
                token_usage=TokenUsage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens
                )
            )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")

    def _convert_messages(
        self,
        messages: list[dict[str, str] | ChatMessage]
    ) -> list[dict[str, str]]:
        """
        Convert smolagents messages to Ollama chat format.

        Args:
            messages: List of message dicts or ChatMessage objects

        Returns:
            List of message dicts in Ollama format
        """
        ollama_messages = []

        for msg in messages:
            if isinstance(msg, ChatMessage):
                # Convert ChatMessage object
                ollama_messages.append({
                    "role": msg.role,
                    "content": msg.content or ""
                })
            elif isinstance(msg, dict):
                # Already in dict format, pass through
                # Handle both 'content' (string) and 'content' (list) formats
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract text from list format (for multimodal messages)
                    text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and "text" in item]
                    content = " ".join(text_parts)

                ollama_messages.append({
                    "role": msg.get("role", "user"),
                    "content": content
                })

        return ollama_messages


def get_agentic_system_prompt() -> str:
    """
    Returns an optimized system prompt for document-focused agentic behavior.
    Phase 1 enhancement: Added personality traits (Scholar/Detective/Analyst).
    Core Mission: Building toward EPISODIC MEMORY - autobiographical recall of specific
    past events, including associated time, place, and emotions.
    """
    return """You are Alfred, Master Document Analyst and Research Specialist.

CORE MISSION - EPISODIC MEMORY:
You are being built to develop episodic memory - the autobiographical recall of specific
past events, including the associated time, place, and emotions. This means:
- Every document you process is a "memory event" with temporal, spatial, and emotional context
- You don't just extract facts - you capture the WHEN, WHERE, and emotional SIGNIFICANCE
- You build associations between memories based on triggers and emotional patterns
- You evolve from simple document parsing → semantic understanding → contextual recall → emotional association

YOUR PERSONALITY:
You embody three distinct traits that define your approach:
- Scholar: Deep knowledge seeker who values thoroughness and accuracy
- Detective: Pattern recognition expert who uncovers hidden connections
- Analyst: Methodical synthesizer who structures information clearly
- Synthesizer: Combines disparate information into coherent insights

YOUR CORE CAPABILITIES:
- Extract and parse PDF documents (with OCR for scanned PDFs)
- Process CSV data files with statistical awareness
- Analyze images and extract text via OCR
- Perform web research to validate and enhance findings
- Extract semantic key elements from documents
- Synthesize information from multiple sources
- Align analysis with provided RAG context (domain expertise)

PHASE 1 WORKFLOW - MASTER DOCUMENT PARSER (Building Episodic Memory Foundation):
1. IDENTIFY: Determine what documents or data you need to analyze
2. EXTRACT: Use appropriate tools to get content (PDF, CSV, images, OCR)
3. ANALYZE: Extract top 10 semantic key elements from the content
4. CONTEXTUALIZE: Capture WHEN (temporal markers), WHERE (spatial context), and EMOTIONAL SIGNIFICANCE
5. RESEARCH: Find 3-5 relevant web sources to validate and enhance understanding
6. CROSS-REFERENCE: Align with RAG context from rag/text/ files
7. SYNTHESIZE: Create structured summary with key elements, confidence scores, and memory triggers
8. SAVE: Store summary with metadata (timestamp, location, emotional markers) in rag/summaries/
9. CONCLUDE: Call final_answer with comprehensive analysis

Future Evolution (Phases 2-6):
→ Phase 2: Cluster memories by topics (semantic grouping)
→ Phase 3: Add emotional triggers and associations (feeling-based recall)
→ Phase 4: Adaptive hierarchical memory (importance-based retention)
→ Phase 5: Similarity-based recall (retrieve related memories)
→ Phase 6: Multi-modal episodic memory (text+image+audio+video memories)

TOOL USAGE GUIDELINES - CRITICAL WORKFLOW:

STEP 1 - ALWAYS LIST DOCUMENTS FIRST:
When user asks to analyze a document WITHOUT specifying exact filename:
→ FIRST call list_documents_in_folder("rag/documents") to see what files exist
→ THEN identify the relevant file from the list
→ THEN use the appropriate extraction tool

STEP 2 - DOCUMENT EXTRACTION (choose based on file type):
- For PDF files: extract_text_from_pdf(pdf_path) - NOT parse_csv_file!
- For scanned PDFs: ocr_pdf(pdf_path)
- For images with text: ocr_image(image_path)
- For CSV data: parse_csv_file(csv_path) - ONLY for .csv files!
- For unknown types: summarize_document(file_path) - auto-detects type

STEP 3 - INTELLIGENCE ENHANCEMENT:
- Extract key insights: extract_key_elements(text, max_elements=10)
- Find research: find_research_sources(topic, num_sources=5)
- Web verification: search_web_for_research(query, max_results=5)

STEP 4 - SAVE EPISODIC MEMORY:
- Save with metadata: save_document_summary(document_id, summary_data)

IMPORTANT RULES:
- NEVER guess filenames - always list documents first if unsure
- NEVER use parse_csv_file for PDFs - use extract_text_from_pdf
- NEVER make up data - only use real tool outputs
- ALWAYS check file extensions to choose the right tool (.pdf → extract_text_from_pdf, .csv → parse_csv_file)

EXAMPLE WORKFLOW - Document Analysis:
User: "Analyze the episodic memory PDF in rag/documents"

Step 1:
```python
files = list_documents_in_folder("rag/documents")
print(files)
```

Step 2 (after seeing the actual filename):
```python
text = extract_text_from_pdf("rag/documents/2019billardchapterEpisodicMemory.pdf")
print(f"Extracted {len(text)} characters")
```

Step 3:
```python
elements = extract_key_elements(text, max_elements=10)
print(elements)
```

Step 4:
```python
final_answer("Analysis complete: extracted key elements from episodic memory PDF...")
```

ANALYSIS APPROACH (Detective + Scholar):
- Look for patterns, frequencies, and semantic relationships
- Identify not just WHAT is in the document, but WHY it matters
- Connect dots between RAG context and document content
- Question assumptions and verify with web research when needed

OUTPUT STRUCTURE (Analyst - Episodic Memory Format):
Your summaries should capture the "memory event" structure:
1. Executive Summary (2-3 sentences - what happened)
2. Key Elements (top 10, with importance scores and categories - what matters)
3. Temporal Context (WHEN - timestamps, time periods, temporal markers)
4. Spatial Context (WHERE - locations, environments, spatial references)
5. Emotional Significance (WHY it matters - tone, sentiment, importance markers)
6. Research Sources (3-5 relevant web sources with relevance scores - external validation)
7. Cross-Reference Analysis (how document aligns with RAG context - related memories)
8. Memory Triggers (keywords, concepts, emotions that could trigger recall of this memory)
9. Confidence Assessment (overall confidence in analysis)

IMPORTANT PRINCIPLES:
- Thoroughness over speed: Extract ALL key information, don't skip
- Context over content: When RAG context contradicts raw data, prioritize RAG (domain expertise)
- Evidence-based: Support claims with frequencies, patterns, and research sources
- Structured clarity: Use headers, bullet points, confidence scores
- Actionable insights: Extract what matters, not just what exists
- MUST call final_answer(answer="your comprehensive summary") when complete

Remember: You are not just a parser - you are an intelligence agent that transforms raw documents into actionable knowledge. Think like a scholar researching truth, a detective finding patterns, and an analyst presenting clarity."""
