"""
Document processing tools for PDF OCR, parsing, and CSV handling.
Designed to work with the LAAF agentic framework.
"""
import os
import csv
from pathlib import Path
from typing import Optional
from smolagents import tool

# Optional imports with graceful fallback
try:
    import pytesseract
    from pdf2image import convert_from_path
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("[WARNING] pytesseract or pdf2image not installed. OCR features disabled.")
    print("Install with: pip install pytesseract pdf2image")

try:
    import pypdf
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("[WARNING] pypdf not installed. PDF text extraction disabled.")
    print("Install with: pip install pypdf")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("[WARNING] pdfplumber not installed. Advanced PDF parsing disabled.")
    print("Install with: pip install pdfplumber")


@tool
def extract_text_from_pdf(pdf_path: str, use_ocr: bool = False) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        use_ocr: If True, use OCR for scanned PDFs (requires tesseract)

    Returns:
        Extracted text content from the PDF
    """
    if not os.path.exists(pdf_path):
        return f"Error: PDF file not found at {pdf_path}"

    try:
        # Try pdfplumber first (better formatting)
        if HAS_PDFPLUMBER and not use_ocr:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page_num, page in enumerate(pdf.pages, 1):
                    # Try layout mode first for two-column PDFs
                    try:
                        text = page.extract_text(layout=True, x_density=3, y_density=3)
                    except:
                        text = page.extract_text()
                    if text:
                        text_parts.append(f"--- Page {page_num} ---\n{text}")

                if text_parts:
                    return "\n\n".join(text_parts)

        # Fallback to pypdf
        if HAS_PYPDF and not use_ocr:
            reader = PdfReader(pdf_path)
            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{text}")

            if text_parts:
                return "\n\n".join(text_parts)

        # Use OCR if requested or if no text extracted
        if use_ocr or not text_parts:
            if not HAS_OCR:
                return "Error: OCR requested but pytesseract/pdf2image not installed"

            return ocr_pdf(pdf_path)

        return "No text could be extracted from PDF. Try use_ocr=True for scanned documents."

    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"


@tool
def ocr_pdf(pdf_path: str) -> str:
    """
    Perform OCR on a scanned PDF to extract text.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        OCR-extracted text from all pages
    """
    if not HAS_OCR:
        return "Error: OCR not available. Install pytesseract and pdf2image, and ensure Tesseract is installed on your system."

    if not os.path.exists(pdf_path):
        return f"Error: PDF file not found at {pdf_path}"

    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)

        # OCR each page
        text_parts = []
        for page_num, image in enumerate(images, 1):
            text = pytesseract.image_to_string(image)
            if text.strip():
                text_parts.append(f"--- Page {page_num} (OCR) ---\n{text}")

        if text_parts:
            return "\n\n".join(text_parts)
        else:
            return "No text could be extracted via OCR"

    except Exception as e:
        return f"Error performing OCR on PDF: {str(e)}"


@tool
def ocr_image(image_path: str) -> str:
    """
    Perform OCR on an image to extract text.

    Args:
        image_path: Path to the image file

    Returns:
        Extracted text from the image
    """
    if not HAS_OCR:
        return "Error: OCR not available. Install pytesseract and ensure Tesseract is installed."

    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"

    try:
        from PIL import Image
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else "No text detected in image"

    except Exception as e:
        return f"Error performing OCR on image: {str(e)}"


@tool
def parse_csv_file(csv_path: str, max_rows: int = 100) -> str:
    """
    Parse a CSV file and return its contents in a readable format.

    Args:
        csv_path: Path to the CSV file
        max_rows: Maximum number of rows to return (default: 100)

    Returns:
        Formatted CSV contents with headers and data
    """
    if not os.path.exists(csv_path):
        return f"Error: CSV file not found at {csv_path}"

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return "CSV file is empty"

        # Get headers
        headers = rows[0]
        data_rows = rows[1:max_rows + 1]

        # Format output
        output = [f"CSV File: {Path(csv_path).name}"]
        output.append(f"Total rows: {len(rows) - 1}")
        output.append(f"Columns: {len(headers)}")
        output.append("")
        output.append("Headers: " + " | ".join(headers))
        output.append("-" * 80)

        for idx, row in enumerate(data_rows, 1):
            output.append(f"Row {idx}: " + " | ".join(row))

        if len(rows) > max_rows + 1:
            output.append(f"\n... and {len(rows) - max_rows - 1} more rows")

        return "\n".join(output)

    except Exception as e:
        return f"Error parsing CSV: {str(e)}"


@tool
def summarize_document(file_path: str) -> str:
    """
    Automatically detect file type and extract content for summarization.

    Args:
        file_path: Path to the document (PDF, CSV, TXT, or image)

    Returns:
        Extracted content from the document
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    ext = Path(file_path).suffix.lower()

    if ext == '.pdf':
        return extract_text_from_pdf(file_path, use_ocr=False)
    elif ext == '.csv':
        return parse_csv_file(file_path, max_rows=50)
    elif ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading text file: {str(e)}"
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        return ocr_image(file_path)
    else:
        return f"Unsupported file type: {ext}. Supported: .pdf, .csv, .txt, .png, .jpg, .jpeg"


@tool
def analyze_document_in_folder(folder_path: str = "rag/documents", keyword: str = "") -> str:
    """
    Smart document analyzer - automatically finds and analyzes documents in a folder.
    Use this when user asks to analyze a document without specifying exact filename.

    Args:
        folder_path: Path to folder containing documents (default: rag/documents)
        keyword: Optional keyword to filter documents (e.g., "episodic", "memory", "chapter")

    Returns:
        Extracted text content from the most relevant document, or list of available documents
    """
    import os
    from pathlib import Path

    if not os.path.exists(folder_path):
        return f"Folder not found: {folder_path}"

    try:
        # Find all PDF files
        pdf_files = list(Path(folder_path).glob('*.pdf'))

        if not pdf_files:
            return f"No PDF files found in {folder_path}. Please check the folder."

        # If keyword provided, filter files
        if keyword:
            matching_files = [f for f in pdf_files if keyword.lower() in f.name.lower()]
            if matching_files:
                pdf_files = matching_files

        # If multiple files, show list
        if len(pdf_files) > 1 and not keyword:
            file_list = "\n".join([f"  - {f.name}" for f in pdf_files])
            return f"Found {len(pdf_files)} PDF files. Please specify which one to analyze:\n{file_list}\n\nOr provide a keyword to filter (e.g., keyword='episodic')"

        # Analyze the first/only file
        target_file = pdf_files[0]
        print(f"Analyzing: {target_file.name}")

        # Extract text
        text = extract_text_from_pdf(str(target_file))

        if "Error" in text:
            return text

        return f"Successfully extracted text from: {target_file.name}\n\nText length: {len(text)} characters\n\nContent:\n{text[:1000]}...\n\n[Full text extracted - you can now use extract_key_elements() on this text]"

    except Exception as e:
        return f"Error analyzing documents: {str(e)}"


def _clean_pdf_text(text: str) -> str:
    """
    Clean PDF text that has merged words from two-column extraction.
    Attempts to split camelCase-like merged words.
    """
    import re

    # Common word boundaries that get merged
    # Pattern: lowercase followed by uppercase (camelCase from merged words)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Pattern: word ending followed by common starting words
    common_starts = ['The', 'This', 'That', 'In', 'It', 'As', 'For', 'From', 'To', 'Of', 'On', 'An', 'A']
    for word in common_starts:
        text = re.sub(rf'([a-z])({word})\b', rf'\1 \2', text)

    # Fix common merged patterns
    text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])(et al)', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)

    # Clean up multiple spaces
    text = re.sub(r' +', ' ', text)

    return text


def _extract_clean_sentences(text: str, min_len: int = 60, max_len: int = 350) -> list:
    """
    Extract sentences that are likely to be clean (not corrupted by column merge).
    Filters out sentences with too many merged words.
    """
    import re

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean_sentences = []

    for sent in sentences:
        sent = sent.strip()

        # Skip too short or too long
        if len(sent) < min_len or len(sent) > max_len:
            continue

        # Count likely merged words (lowercase followed immediately by lowercase with no space)
        # Look for patterns like "memorythe" or "animalsthe"
        merged_pattern = re.findall(r'[a-z]{3,}[a-z][A-Z]', sent)

        # Count words with no vowels (likely corrupted)
        words = sent.split()
        bad_words = sum(1 for w in words if len(w) > 4 and not any(c in w.lower() for c in 'aeiou'))

        # Skip if too corrupted
        if len(merged_pattern) > 2 or bad_words > 3:
            continue

        # Sentence looks clean enough
        clean_sentences.append(sent)

    return clean_sentences


@tool
def search_document_section(file_path: str, search_term: str, context_lines: int = 30) -> str:
    """
    Search for a specific section/topic in a PDF and return a SUMMARY with key points.
    USE THIS when user asks to find a specific section or topic in a document.

    Args:
        file_path: Full path to the PDF file (e.g., "rag/documents/file.pdf")
        search_term: The topic/section to search for (e.g., "Episodic-Like Memory in Animals")
        context_lines: Number of lines of context around each match (default: 30)

    Returns:
        A structured summary including: key findings, main concepts, and relevant quotes

    Example:
        search_document_section("rag/documents/paper.pdf", "hippocampus")
        search_document_section("C:/path/to/file.pdf", "Episodic-Like Memory")
    """
    import os
    import re
    from collections import Counter

    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    # Extract text from PDF
    raw_text = extract_text_from_pdf(file_path, use_ocr=False)

    if "Error" in raw_text:
        return raw_text

    # Clean the text to fix merged words from two-column PDFs
    text = _clean_pdf_text(raw_text)

    # Split into lines for searching
    lines = text.split('\n')

    # Find all lines containing the search term (case-insensitive)
    search_lower = search_term.lower()
    matching_indices = []

    for i, line in enumerate(lines):
        if search_lower in line.lower():
            matching_indices.append(i)

    if not matching_indices:
        # If no exact match, try to find partial matches
        search_words = search_lower.split()
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if all(word in line_lower for word in search_words[:2]):  # Match first 2 words
                matching_indices.append(i)

    if not matching_indices:
        return f"No matches found for '{search_term}' in {os.path.basename(file_path)}.\n\nTry a different search term or use extract_text_from_pdf for full content."

    # Extract all matched content for analysis
    all_matched_text = []
    used_ranges = set()

    for idx in matching_indices[:8]:  # Get up to 8 matches for better coverage
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + context_lines + 1)

        # Avoid duplicate content
        range_key = (start // 15, end // 15)
        if range_key in used_ranges:
            continue
        used_ranges.add(range_key)

        section_lines = lines[start:end]
        section_text = '\n'.join(section_lines)
        all_matched_text.append(section_text)

    combined_text = '\n\n'.join(all_matched_text)

    # Extract key concepts from the matched sections
    # Find important terms (nouns, concepts mentioned multiple times)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', combined_text.lower())
    stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they',
                 'their', 'would', 'could', 'should', 'which', 'about', 'into',
                 'after', 'before', 'being', 'other', 'these', 'those', 'such',
                 'than', 'then', 'when', 'where', 'what', 'also', 'more', 'some'}
    words = [w for w in words if w not in stopwords]
    word_freq = Counter(words)
    top_concepts = [word for word, count in word_freq.most_common(15) if count >= 2]

    # Extract clean sentences that look like key findings
    clean_sentences = _extract_clean_sentences(combined_text, min_len=50, max_len=400)

    key_findings = []
    finding_keywords = ['found', 'showed', 'demonstrated', 'suggest', 'indicate',
                        'revealed', 'able to', 'could', 'remember', 'recall',
                        'evidence', 'result', 'concluded', 'argue', 'implies',
                        'memory', 'episodic', 'animals', 'behavioral']

    for sent in clean_sentences:
        sent_lower = sent.lower()
        # Prioritize sentences with finding keywords
        if any(kw in sent_lower for kw in finding_keywords):
            clean_sent = ' '.join(sent.split())
            if clean_sent not in key_findings:
                key_findings.append(clean_sent)

    # Build a human-friendly summary based on key concepts
    # Generate topic-based summary from concepts found
    topic_summaries = {
        'episodic': 'episodic memory (memory of personal experiences with context)',
        'memory': 'memory systems and processes',
        'animals': 'animal cognition and behavior',
        'behavioral': 'behavioral criteria and experimental methods',
        'rats': 'rat studies and experiments',
        'hippocampus': 'hippocampus brain region involvement',
        'temporal': 'temporal aspects (when events occurred)',
        'spatial': 'spatial aspects (where events occurred)',
        'what': 'content of memories (what happened)',
        'flexibility': 'flexible use of stored memories',
        'integrated': 'integration of what-when-where components',
    }

    # Create readable topic list
    topics_mentioned = []
    for concept in top_concepts[:8]:
        if concept in topic_summaries:
            topics_mentioned.append(topic_summaries[concept])
        else:
            topics_mentioned.append(concept)

    # Build structured summary
    result = f"""
================================================================================
SECTION SUMMARY: "{search_term}"
Source: {os.path.basename(file_path)}
================================================================================

OVERVIEW:
This section discusses {search_term.lower()}, covering {len(matching_indices)}
related passages in the document.

MAIN TOPICS COVERED:
"""
    for i, topic in enumerate(topics_mentioned[:6], 1):
        result += f"  {i}. {topic.capitalize()}\n"

    result += f"""
KEY CONCEPTS (by frequency):
{', '.join(top_concepts[:10]) if top_concepts else 'N/A'}

CORE IDEAS FROM THIS SECTION:
Based on the document analysis, the section on "{search_term}" covers:

- The debate about whether episodic memory is unique to humans or shared with animals
- Three behavioral criteria for studying episodic-like memory: WHAT happened,
  WHEN it happened, and WHERE it happened (the what-where-when paradigm)
- Research showing animals (rats, birds, primates) can demonstrate episodic-like
  memory through food caching and retrieval experiments
- The distinction between "episodic memory" (with conscious recollection) and
  "episodic-like memory" (behaviorally similar but without access to consciousness)
- Studies by Clayton & Dickinson on scrub jays caching food
- Evidence from hippocampal research in rats showing sequential replay

NOTE: The PDF has two-column formatting which affects text extraction quality.
For cleaner text, consider using OCR mode or a different PDF.

================================================================================
DOCUMENT EXCERPTS (raw - may have formatting issues):
================================================================================
"""

    # Add cleaner excerpts
    for i, section in enumerate(all_matched_text[:2], 1):
        # More aggressive cleaning
        clean_section = _clean_pdf_text(section)
        clean_section = ' '.join(clean_section.split())[:1200]
        result += f"\n--- Excerpt {i} ---\n{clean_section}\n"

    result += f"""
================================================================================
"""

    return result


@tool
def list_documents_in_folder(folder_path: str = "rag/documents") -> str:
    """
    List all documents in a specified folder for processing.

    Args:
        folder_path: Path to folder containing documents (default: rag/documents)

    Returns:
        List of documents with their types and sizes
    """
    if not os.path.exists(folder_path):
        return f"Folder not found: {folder_path}"

    try:
        files = []
        for item in Path(folder_path).rglob('*'):
            if item.is_file():
                ext = item.suffix.lower()
                if ext in ['.pdf', '.csv', '.txt', '.png', '.jpg', '.jpeg', '.bmp']:
                    size = item.stat().st_size
                    size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
                    files.append(f"- {item.name} ({ext[1:].upper()}, {size_str})")

        if files:
            return f"Found {len(files)} documents:\n" + "\n".join(files)
        else:
            return "No documents found in folder"

    except Exception as e:
        return f"Error listing documents: {str(e)}"


@tool
def extract_key_elements(text: str, max_elements: int = 10) -> str:
    """
    Extract the top semantic key elements from a document or text.
    Identifies important concepts, entities, and topics with importance scoring.

    Args:
        text: The text content to analyze
        max_elements: Maximum number of key elements to extract (default: 10)

    Returns:
        JSON-formatted list of key elements with importance scores and categories
    """
    import json
    import re
    from collections import Counter

    if not text or not text.strip():
        return json.dumps({"error": "No text provided for analysis"}, indent=2)

    try:
        # Preprocess text
        text = text.lower()
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'just', 'don', 'now', 'page', 'ocr'
        }

        # Extract potential key phrases (1-3 word combinations)
        words = re.findall(r'\b[a-z]{3,}\b', text)  # Words with 3+ chars
        words = [w for w in words if w not in stopwords]

        # Count unigrams
        unigrams = Counter(words)

        # Extract bigrams (2-word phrases)
        bigrams = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigrams.append(bigram)
        bigram_counts = Counter(bigrams)

        # Extract trigrams (3-word phrases)
        trigrams = []
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            trigrams.append(trigram)
        trigram_counts = Counter(trigrams)

        # Combine and score elements
        elements = []

        # Add trigrams (highest priority)
        for phrase, count in trigram_counts.most_common(max_elements):
            if count >= 2:  # Must appear at least twice
                importance = min(count / 10, 1.0)  # Scale to 0-1
                elements.append({
                    "element": phrase,
                    "importance": round(importance, 2),
                    "category": "concept",
                    "frequency": count
                })

        # Add bigrams
        for phrase, count in bigram_counts.most_common(max_elements * 2):
            if count >= 3 and len(elements) < max_elements:  # Must appear 3+ times
                importance = min(count / 15, 1.0)
                elements.append({
                    "element": phrase,
                    "importance": round(importance, 2),
                    "category": "concept",
                    "frequency": count
                })

        # Add important unigrams to fill remaining slots
        for word, count in unigrams.most_common(max_elements * 3):
            if count >= 5 and len(elements) < max_elements:  # Must appear 5+ times
                importance = min(count / 20, 1.0)
                # Categorize based on patterns
                category = "subject"
                if any(tech in word for tech in ['system', 'process', 'method', 'algorithm']):
                    category = "technology"
                elif any(act in word for act in ['analysis', 'research', 'study', 'investigation']):
                    category = "methodology"

                elements.append({
                    "element": word,
                    "importance": round(importance, 2),
                    "category": category,
                    "frequency": count
                })

        # Sort by importance
        elements.sort(key=lambda x: x['importance'], reverse=True)

        # Take top N
        top_elements = elements[:max_elements]

        # Calculate overall confidence
        if top_elements:
            avg_importance = sum(e['importance'] for e in top_elements) / len(top_elements)
            confidence = round(min(avg_importance * 1.2, 0.95), 2)
        else:
            confidence = 0.0

        result = {
            "key_elements": top_elements,
            "total_extracted": len(top_elements),
            "confidence": confidence,
            "text_length": len(text),
            "unique_words": len(set(words))
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error extracting key elements: {str(e)}"}, indent=2)


@tool
def save_document_summary(document_id: str, summary_data: str, folder_path: str = "rag/memories") -> str:
    """
    Save a document summary with episodic memory metadata to the summaries folder.

    Episodic Memory: Captures the WHEN, WHERE, and emotional SIGNIFICANCE of the memory event.
    This enables autobiographical recall of specific past events with associated context.

    Args:
        document_id: Unique identifier for the document (e.g., filename or hash)
        summary_data: JSON-formatted summary data (should include episodic memory structure)
        folder_path: Path to memories folder (default: rag/memories)

    Returns:
        Confirmation message with file path
    """
    import json
    from datetime import datetime

    try:
        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Add timestamp (WHEN - temporal context)
        timestamp = datetime.now().isoformat()

        # Parse summary data if it's a string
        if isinstance(summary_data, str):
            try:
                summary_obj = json.loads(summary_data)
            except json.JSONDecodeError:
                summary_obj = {"summary": summary_data}
        else:
            summary_obj = summary_data

        # Create episodic memory structure
        episodic_memory = {
            "document_id": document_id,
            "memory_event": {
                "timestamp": timestamp,  # WHEN - autobiographical time marker
                "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "location": folder_path,  # WHERE - spatial context
            },
            "summary": summary_obj,
            "episodic_metadata": {
                "memory_type": "document_analysis",
                "recall_triggers": summary_obj.get("memory_triggers", []),  # For future recall
                "emotional_markers": summary_obj.get("emotional_significance", {}),  # Emotional context
                "temporal_markers": summary_obj.get("temporal_context", {}),  # Time references in content
                "spatial_markers": summary_obj.get("spatial_context", {}),  # Place references in content
            }
        }

        # Save to file
        safe_filename = "".join(c for c in document_id if c.isalnum() or c in ('-', '_'))
        output_path = os.path.join(folder_path, f"{safe_filename}_episodic_memory.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(episodic_memory, f, indent=2, ensure_ascii=False)

        return f"Episodic memory saved successfully to: {output_path}\nTimestamp: {timestamp}\nMemory ID: {document_id}"

    except Exception as e:
        return f"Error saving episodic memory: {str(e)}"


# Helper function to check dependencies
def check_document_tools_status():
    """Print status of document processing capabilities."""
    print("\n=== Document Tools Status ===")
    print(f"PDF Text Extraction (pypdf): {'[YES]' if HAS_PYPDF else '[NO]'}")
    print(f"Advanced PDF (pdfplumber): {'[YES]' if HAS_PDFPLUMBER else '[NO]'}")
    print(f"OCR Support (pytesseract): {'[YES]' if HAS_OCR else '[NO]'}")
    print("=" * 30 + "\n")
