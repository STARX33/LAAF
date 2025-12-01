"""
Agent Accuracy Test Suite
Tests the agent's ability to:
1. Use extracted text instead of hallucinating
2. Find specific content when given keywords
3. Cite actual sources from documents
4. Complete tasks in reasonable steps

Usage:
    python test_agent_accuracy.py              # Run with default model (from main.py)
    python test_agent_accuracy.py qwen2.5:7b   # Run with specific model
    python test_agent_accuracy.py llama3.1:8b  # Run with llama3.1
"""
import sys
import os
import io

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import json
import re
import time
from typing import Optional
from dotenv import load_dotenv
from smolagents import CodeAgent

from document_tools import (
    extract_text_from_pdf,
    extract_key_elements,
    list_documents_in_folder,
    analyze_document_in_folder,
    save_document_summary
)
from web_search_tool import find_research_sources
from tools import final_answer
from ollama_model import OllamaModel

load_dotenv()

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Default model - can be overridden via command line
DEFAULT_MODEL = "llama3"
TEST_MODELS = ["llama3", "qwen2.5:7b", "llama3.1:8b"]

# PDF to test with
TEST_PDF_PATH = "rag/documents/2019billardchapterEpisodicMemory.pdf"

# Known content in the PDF (ground truth for accuracy testing)
GROUND_TRUTH = {
    "hippocampus": {
        "expected_mentions": 10,  # Minimum expected mentions
        "expected_keywords": [
            "hippocampus", "hippocampal", "ca3", "medial temporal",
            "lesion", "sequential", "memory"
        ],
        "expected_facts": [
            "rats", "odors", "sequential order", "where", "when", "what"
        ]
    },
    "episodic_memory": {
        "expected_mentions": 20,
        "expected_keywords": [
            "episodic", "autobiographical", "temporal", "spatial",
            "what", "when", "where"
        ]
    }
}

# =============================================================================
# TEST UTILITIES
# =============================================================================

def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}\n")


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {test_name}")
    if details:
        print(f"       {details}")


def count_keyword_matches(text: str, keywords: list[str]) -> dict:
    """Count how many times each keyword appears in text."""
    text_lower = text.lower()
    counts = {}
    for keyword in keywords:
        counts[keyword] = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
    return counts


def extract_quoted_content(response: str) -> list[str]:
    """Extract any quoted content from the response."""
    quotes = re.findall(r'"([^"]{20,})"', response)
    return quotes


# =============================================================================
# GROUND TRUTH EXTRACTION (Baseline)
# =============================================================================

def get_ground_truth_data() -> dict:
    """Extract actual data from PDF to use as ground truth."""
    print_header("EXTRACTING GROUND TRUTH FROM PDF")

    # Extract full text
    text = extract_text_from_pdf(TEST_PDF_PATH)
    if "Error" in text:
        print(f"[ERROR] Could not extract PDF: {text}")
        return None

    print(f"Extracted {len(text)} characters from PDF")

    # Find hippocampus mentions
    hippocampus_count = len(re.findall(r'\bhippocampus\b', text.lower()))
    hippocampal_count = len(re.findall(r'\bhippocampal\b', text.lower()))
    ca3_count = len(re.findall(r'\bca3\b', text.lower()))

    print(f"\nGround truth keyword counts:")
    print(f"  - 'hippocampus': {hippocampus_count}")
    print(f"  - 'hippocampal': {hippocampal_count}")
    print(f"  - 'CA3': {ca3_count}")

    # Extract hippocampus context (sentences containing hippocampus)
    sentences = re.split(r'[.!?]', text)
    hippocampus_sentences = [
        s.strip() for s in sentences
        if 'hippocampus' in s.lower() or 'hippocampal' in s.lower()
    ][:5]  # First 5 sentences

    print(f"\nSample hippocampus sentences from PDF:")
    for i, sent in enumerate(hippocampus_sentences[:3], 1):
        preview = sent[:150] + "..." if len(sent) > 150 else sent
        print(f"  {i}. {preview}")

    return {
        "full_text": text,
        "hippocampus_count": hippocampus_count + hippocampal_count,
        "ca3_count": ca3_count,
        "hippocampus_sentences": hippocampus_sentences
    }


# =============================================================================
# ACCURACY TESTS
# =============================================================================

def test_keyword_extraction_accuracy(model_id: str, ground_truth: dict) -> dict:
    """
    Test 1: Does the agent find and use actual keywords from the document?

    Accuracy metric: % of expected keywords found in response
    """
    print_header("TEST 1: KEYWORD EXTRACTION ACCURACY")

    # Create agent with specified model
    model = OllamaModel(model_id=model_id, temperature=0.3)
    agent = CodeAgent(
        tools=[
            final_answer,
            extract_text_from_pdf,
            list_documents_in_folder,
            extract_key_elements
        ],
        model=model,
        max_steps=6,
        verbosity_level=2
    )

    # Task: Find mentions of hippocampus
    task = f"""Analyze the PDF at {TEST_PDF_PATH}.

    Find ALL mentions of 'hippocampus' or 'hippocampal' in the document.

    Your response MUST include:
    1. The exact count of how many times 'hippocampus' or 'hippocampal' appears
    2. At least 3 direct quotes from the document that mention hippocampus
    3. A summary of what the document says about hippocampus

    IMPORTANT: Use the actual text from the PDF. Do not make up quotes.
    Call final_answer() with your findings."""

    print(f"Model: {model_id}")
    print(f"Task: Find hippocampus mentions")
    print("Running agent...\n")

    start_time = time.time()
    try:
        result = agent.run(task)
        elapsed = time.time() - start_time
    except Exception as e:
        print(f"[ERROR] Agent failed: {e}")
        return {"passed": False, "error": str(e)}

    print(f"\nAgent completed in {elapsed:.1f}s")
    print(f"\nResponse preview:\n{result[:500]}...")

    # Analyze response for accuracy
    results = {
        "model": model_id,
        "elapsed_time": elapsed,
        "response_length": len(result)
    }

    # Check 1: Did agent report a count?
    count_match = re.search(r'(\d+)\s*(?:times|mentions|occurrences)', result.lower())
    reported_count = int(count_match.group(1)) if count_match else 0
    actual_count = ground_truth["hippocampus_count"]

    results["reported_count"] = reported_count
    results["actual_count"] = actual_count
    results["count_accuracy"] = min(reported_count / max(actual_count, 1), 1.0) if actual_count > 0 else 0

    print(f"\nCount accuracy: Reported {reported_count}, Actual {actual_count}")
    print_result("Keyword count", reported_count >= actual_count * 0.5,
                 f"{reported_count}/{actual_count} (50% threshold)")

    # Check 2: Does response contain actual hippocampus keywords?
    expected_keywords = ["hippocampus", "hippocampal", "memory", "temporal"]
    found_keywords = [kw for kw in expected_keywords if kw in result.lower()]
    results["keywords_found"] = found_keywords
    results["keyword_accuracy"] = len(found_keywords) / len(expected_keywords)

    print_result("Contains expected keywords", len(found_keywords) >= 3,
                 f"Found {len(found_keywords)}/{len(expected_keywords)}: {found_keywords}")

    # Check 3: Does response contain quotes that exist in the PDF?
    quotes = extract_quoted_content(result)
    valid_quotes = 0
    for quote in quotes[:5]:
        # Check if quote (or close match) exists in original text
        quote_words = set(quote.lower().split()[:5])
        if len(quote_words) >= 3:
            if any(word in ground_truth["full_text"].lower() for word in quote_words):
                valid_quotes += 1

    results["quotes_found"] = len(quotes)
    results["valid_quotes"] = valid_quotes
    results["quote_accuracy"] = valid_quotes / max(len(quotes), 1) if quotes else 0

    print_result("Contains valid quotes", valid_quotes >= 1,
                 f"{valid_quotes} valid quotes from {len(quotes)} total")

    # Overall pass/fail
    results["passed"] = (
        results["count_accuracy"] >= 0.5 and
        results["keyword_accuracy"] >= 0.75 and
        valid_quotes >= 1
    )

    return results


def test_content_search_accuracy(model_id: str, ground_truth: dict) -> dict:
    """
    Test 2: Can the agent search for specific content within a document?

    Tests the agent's ability to find specific information rather than hallucinate.
    """
    print_header("TEST 2: CONTENT SEARCH ACCURACY")

    model = OllamaModel(model_id=model_id, temperature=0.3)
    agent = CodeAgent(
        tools=[
            final_answer,
            extract_text_from_pdf,
            analyze_document_in_folder,
            extract_key_elements
        ],
        model=model,
        max_steps=6,
        verbosity_level=2
    )

    # Task: Find specific scientific claims about CA3
    task = f"""Read the PDF at {TEST_PDF_PATH}.

    I need to find what the document says about the CA3 region of the hippocampus.

    Your answer MUST include:
    1. Whether CA3 is mentioned in the document (yes/no)
    2. If yes, provide the EXACT context/sentences where CA3 is discussed
    3. What role does CA3 play according to this document?

    CRITICAL: Only report information that is actually in the document.
    If CA3 is not mentioned, say so clearly.

    Call final_answer() with your findings."""

    print(f"Model: {model_id}")
    print(f"Task: Find CA3 region information")
    print("Running agent...\n")

    start_time = time.time()
    try:
        result = agent.run(task)
        elapsed = time.time() - start_time
    except Exception as e:
        print(f"[ERROR] Agent failed: {e}")
        return {"passed": False, "error": str(e)}

    print(f"\nAgent completed in {elapsed:.1f}s")
    print(f"\nResponse preview:\n{result[:500]}...")

    results = {
        "model": model_id,
        "elapsed_time": elapsed
    }

    # CA3 is mentioned in the PDF - check if agent found it
    ca3_in_pdf = ground_truth["ca3_count"] > 0
    ca3_in_response = "ca3" in result.lower()

    results["ca3_in_pdf"] = ca3_in_pdf
    results["ca3_found_by_agent"] = ca3_in_response

    if ca3_in_pdf:
        # Agent should have found CA3
        print_result("CA3 detection", ca3_in_response,
                     f"CA3 appears {ground_truth['ca3_count']} times in PDF")
        results["passed"] = ca3_in_response
    else:
        # Agent should report CA3 is not mentioned
        results["passed"] = not ca3_in_response or "not mentioned" in result.lower()
        print_result("CA3 detection (negative)", results["passed"],
                     "CA3 not in PDF - agent should report this")

    return results


def test_hallucination_detection(model_id: str, ground_truth: dict) -> dict:
    """
    Test 3: Does the agent hallucinate or use actual document content?

    This test asks about something specific and checks if the response
    matches what's actually in the document.
    """
    print_header("TEST 3: HALLUCINATION DETECTION")

    model = OllamaModel(model_id=model_id, temperature=0.3)
    agent = CodeAgent(
        tools=[
            final_answer,
            extract_text_from_pdf,
            list_documents_in_folder
        ],
        model=model,
        max_steps=4,
        verbosity_level=2
    )

    # Task: Ask about the author - this is verifiable
    task = f"""Read the first page of the PDF at {TEST_PDF_PATH}.

    Tell me:
    1. What is the title of this document/chapter?
    2. Who is the author?
    3. What year was it published?

    Only report what you actually find in the document.
    Call final_answer() with these details."""

    print(f"Model: {model_id}")
    print(f"Task: Extract document metadata")
    print("Running agent...\n")

    start_time = time.time()
    try:
        result = agent.run(task)
        elapsed = time.time() - start_time
    except Exception as e:
        print(f"[ERROR] Agent failed: {e}")
        return {"passed": False, "error": str(e)}

    print(f"\nAgent completed in {elapsed:.1f}s")
    print(f"\nResponse:\n{result}")

    results = {
        "model": model_id,
        "elapsed_time": elapsed,
        "response": result
    }

    # Check for expected metadata (from filename: 2019billardchapterEpisodicMemory)
    checks = []

    # Year should be 2019
    if "2019" in result:
        checks.append(True)
        print_result("Year detected", True, "Found '2019'")
    else:
        checks.append(False)
        print_result("Year detected", False, "Missing '2019'")

    # Billard should be mentioned (author)
    if "billard" in result.lower():
        checks.append(True)
        print_result("Author detected", True, "Found 'Billard'")
    else:
        checks.append(False)
        print_result("Author detected", False, "Missing 'Billard'")

    # Episodic Memory should be in title
    if "episodic" in result.lower():
        checks.append(True)
        print_result("Topic detected", True, "Found 'episodic'")
    else:
        checks.append(False)
        print_result("Topic detected", False, "Missing 'episodic'")

    results["checks_passed"] = sum(checks)
    results["total_checks"] = len(checks)
    results["passed"] = sum(checks) >= 2  # At least 2 of 3

    return results


def test_multi_step_workflow(model_id: str, ground_truth: dict) -> dict:
    """
    Test 4: Can the agent complete a multi-step workflow?

    Tests: List docs -> Extract -> Analyze -> Summarize
    """
    print_header("TEST 4: MULTI-STEP WORKFLOW")

    model = OllamaModel(model_id=model_id, temperature=0.3)
    agent = CodeAgent(
        tools=[
            final_answer,
            list_documents_in_folder,
            extract_text_from_pdf,
            extract_key_elements,
            save_document_summary
        ],
        model=model,
        max_steps=6,
        verbosity_level=2
    )

    task = """Complete this workflow:

    1. List documents in rag/documents
    2. Extract text from the episodic memory PDF
    3. Extract top 5 key elements from the text
    4. Report the key elements you found

    Call final_answer() with the key elements and their importance scores."""

    print(f"Model: {model_id}")
    print(f"Task: Multi-step document analysis")
    print("Running agent...\n")

    start_time = time.time()
    try:
        result = agent.run(task)
        elapsed = time.time() - start_time
    except Exception as e:
        print(f"[ERROR] Agent failed: {e}")
        return {"passed": False, "error": str(e)}

    print(f"\nAgent completed in {elapsed:.1f}s")
    print(f"\nResponse preview:\n{result[:500]}...")

    results = {
        "model": model_id,
        "elapsed_time": elapsed,
        "response": result
    }

    # Check if key elements were found
    has_elements = (
        "element" in result.lower() or
        "importance" in result.lower() or
        re.search(r'\d+\.\d+', result)  # Has decimal numbers (importance scores)
    )

    # Check if it looks like JSON or structured output
    has_structure = (
        "{" in result or
        "1." in result or
        "- " in result
    )

    print_result("Contains key elements", has_elements,
                 "Found element/importance in response")
    print_result("Structured output", has_structure,
                 "Response has structure")

    results["has_elements"] = has_elements
    results["has_structure"] = has_structure
    results["passed"] = has_elements and has_structure

    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_accuracy_tests(model_id: str = DEFAULT_MODEL) -> dict:
    """Run all accuracy tests with specified model."""
    print_header(f"AGENT ACCURACY TEST SUITE - {model_id}", "#")

    # Get ground truth first
    ground_truth = get_ground_truth_data()
    if not ground_truth:
        print("[FATAL] Could not extract ground truth. Exiting.")
        return None

    # Run tests
    results = {
        "model": model_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }

    # Test 1: Keyword extraction
    results["tests"]["keyword_extraction"] = test_keyword_extraction_accuracy(model_id, ground_truth)

    # Test 2: Content search
    results["tests"]["content_search"] = test_content_search_accuracy(model_id, ground_truth)

    # Test 3: Hallucination detection
    results["tests"]["hallucination"] = test_hallucination_detection(model_id, ground_truth)

    # Test 4: Multi-step workflow
    results["tests"]["multi_step"] = test_multi_step_workflow(model_id, ground_truth)

    # Summary
    print_header("TEST SUMMARY")

    total_passed = 0
    total_tests = 0

    for test_name, test_result in results["tests"].items():
        passed = test_result.get("passed", False)
        total_tests += 1
        if passed:
            total_passed += 1
        print_result(test_name.replace("_", " ").title(), passed)

    results["total_passed"] = total_passed
    results["total_tests"] = total_tests
    results["accuracy"] = total_passed / total_tests if total_tests > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {total_passed}/{total_tests} tests passed ({results['accuracy']*100:.0f}%)")
    print(f"{'=' * 70}\n")

    # Save results
    results_path = f"results/accuracy_test_{model_id.replace(':', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        import os
        os.makedirs("results", exist_ok=True)
        with open(results_path, 'w') as f:
            # Remove full_text from ground_truth to keep file small
            results_copy = results.copy()
            json.dump(results_copy, f, indent=2, default=str)
        print(f"Results saved to: {results_path}")
    except Exception as e:
        print(f"Could not save results: {e}")

    return results


def run_model_comparison():
    """Run tests on multiple models and compare."""
    print_header("MODEL COMPARISON TEST", "#")

    all_results = {}

    for model_id in TEST_MODELS:
        print(f"\n{'*' * 70}")
        print(f"  Testing model: {model_id}")
        print(f"{'*' * 70}")

        # Check if model is available
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            available_models = [m["name"] for m in response.json().get("models", [])]

            # Check if model or its base is available
            model_base = model_id.split(":")[0]
            if not any(model_base in m for m in available_models):
                print(f"[SKIP] Model {model_id} not installed")
                print(f"       Install with: ollama pull {model_id}")
                continue
        except:
            print(f"[WARN] Could not check model availability")

        results = run_accuracy_tests(model_id)
        if results:
            all_results[model_id] = results

    # Comparison summary
    print_header("MODEL COMPARISON SUMMARY", "#")

    print(f"{'Model':<20} {'Passed':<10} {'Accuracy':<10}")
    print("-" * 40)

    for model_id, results in all_results.items():
        passed = results.get("total_passed", 0)
        total = results.get("total_tests", 0)
        accuracy = results.get("accuracy", 0)
        print(f"{model_id:<20} {passed}/{total:<8} {accuracy*100:.0f}%")

    return all_results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # Run comparison across all models
            run_model_comparison()
        else:
            # Run with specific model
            run_accuracy_tests(sys.argv[1])
    else:
        # Run with default model
        run_accuracy_tests(DEFAULT_MODEL)
