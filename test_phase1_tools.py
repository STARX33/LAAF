"""
Test script for Phase 1: Episodic Memory Foundation
Tests each tool individually to verify functionality.
"""
import sys
from document_tools import (
    extract_text_from_pdf,
    extract_key_elements,
    save_document_summary
)
from web_search_tool import (
    search_web_for_research,
    find_research_sources
)

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def test_pdf_extraction():
    """Test 1: PDF Text Extraction"""
    print_section("TEST 1: PDF TEXT EXTRACTION")

    pdf_path = "rag/documents/2019billardchapterEpisodicMemory.pdf"
    print(f"Extracting text from: {pdf_path}")
    print("Please wait...\n")

    try:
        extracted_text = extract_text_from_pdf(pdf_path)

        # Check if extraction was successful
        if "Error" in extracted_text:
            print("[FAIL] PDF extraction failed!")
            print(extracted_text)
            return None

        print("[PASS] PDF extraction successful!")
        print(f"   Total characters extracted: {len(extracted_text)}")
        print(f"   Total lines: {len(extracted_text.splitlines())}")
        print(f"\n   First 500 characters:")
        print("   " + "-" * 66)
        print("   " + extracted_text[:500].replace("\n", "\n   "))
        print("   " + "-" * 66)

        return extracted_text

    except Exception as e:
        print(f"[ERROR] Exception during PDF extraction: {e}")
        return None

def test_key_elements(text):
    """Test 2: Key Elements Extraction"""
    print_section("TEST 2: KEY ELEMENTS EXTRACTION")

    if not text:
        print("[SKIP] Skipping - no text available from previous test")
        return None

    print("Extracting top 10 semantic key elements...")
    print("Please wait...\n")

    try:
        # Use first 5000 characters for faster testing
        sample_text = text[:5000] if len(text) > 5000 else text
        print(f"Analyzing {len(sample_text)} characters (sample)")

        result = extract_key_elements(sample_text, max_elements=10)

        # Check if extraction was successful
        if "error" in result.lower():
            print("[FAIL] Key elements extraction failed!")
            print(result)
            return None

        print("[PASS] Key elements extraction successful!")
        print("\nResults:")
        print(result)

        return result

    except Exception as e:
        print(f"[ERROR] Exception during key elements extraction: {e}")
        return None

def test_web_research():
    """Test 3: Web Research Tools"""
    print_section("TEST 3: WEB RESEARCH (DuckDuckGo)")

    query = "episodic memory cognitive neuroscience"
    print(f"Search query: '{query}'")
    print("Finding 5 research sources...")
    print("Please wait (this may take 10-15 seconds)...\n")

    try:
        result = find_research_sources(query, num_sources=5)

        # Check if search was successful
        if "Error" in result or "not available" in result:
            print("[FAIL] Web research failed!")
            print(result)
            return None

        print("[PASS] Web research successful!")
        print("\nResults:")
        print(result)

        return result

    except Exception as e:
        print(f"[ERROR] Exception during web research: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_save_summary():
    """Test 4: Episodic Memory Save"""
    print_section("TEST 4: EPISODIC MEMORY SAVE")

    print("Creating test episodic memory structure...")

    test_summary = {
        "executive_summary": "Test summary of episodic memory chapter",
        "key_elements": [
            {"element": "episodic memory", "importance": 0.95, "category": "concept"},
            {"element": "autobiographical recall", "importance": 0.85, "category": "concept"}
        ],
        "temporal_context": {
            "document_date": "2019",
            "time_references": ["past events", "specific moments"]
        },
        "spatial_context": {
            "locations_mentioned": ["hippocampus", "medial temporal lobe"]
        },
        "emotional_significance": {
            "tone": "academic",
            "importance_level": "high",
            "markers": ["memory formation", "cognitive processes"]
        },
        "memory_triggers": ["episodic memory", "autobiographical", "temporal context", "hippocampus"],
        "research_sources": [
            {"title": "Episodic Memory Research", "url": "http://example.com", "relevance": 0.9}
        ]
    }

    import json
    summary_json = json.dumps(test_summary, indent=2)

    try:
        result = save_document_summary(
            document_id="test_2019billardchapter",
            summary_data=summary_json,
            folder_path="rag/memories"
        )

        # Check if save was successful
        if "Error" in result:
            print("[FAIL] Episodic memory save failed!")
            print(result)
            return None

        print("[PASS] Episodic memory save successful!")
        print(result)

        # Try to read back the saved file
        print("\nVerifying saved file...")
        saved_path = "rag/memories/test_2019billardchapter_episodic_memory.json"
        try:
            with open(saved_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            print(f"[PASS] File verified! Keys: {list(saved_data.keys())}")
            print(f"   Memory event timestamp: {saved_data.get('memory_event', {}).get('timestamp', 'N/A')}")
        except Exception as e:
            print(f"[WARN] Could not verify file: {e}")

        return result

    except Exception as e:
        print(f"[ERROR] Exception during save: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all tests sequentially."""
    print("\n" + "#" * 70)
    print("  PHASE 1: EPISODIC MEMORY FOUNDATION - TOOL TESTS")
    print("#" * 70)

    # Test 1: PDF Extraction
    extracted_text = test_pdf_extraction()

    # Test 2: Key Elements (only if PDF extraction succeeded)
    key_elements = test_key_elements(extracted_text)

    # Test 3: Web Research
    research_results = test_web_research()

    # Test 4: Save Episodic Memory
    save_result = test_save_summary()

    # Final Summary
    print_section("TEST SUMMARY")

    results = {
        "PDF Extraction": extracted_text is not None,
        "Key Elements Extraction": key_elements is not None,
        "Web Research": research_results is not None,
        "Episodic Memory Save": save_result is not None
    }

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} - {test_name}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print(f"{'=' * 70}\n")

    if total_passed == total_tests:
        print("SUCCESS! ALL TESTS PASSED! Phase 1 tools are working correctly.")
    else:
        print("WARNING: Some tests failed. Please review errors above.")

    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
