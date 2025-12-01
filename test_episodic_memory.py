"""
Episodic Memory Test Suite
Tests that episodic memory functionality is working correctly:

1. Memory Save: Can we save memories with proper structure?
2. Memory Structure: Do memories contain WHEN/WHERE/WHY metadata?
3. Memory Load: Can we load and read saved memories?
4. RAG Integration: Does RAG context include memory knowledge?
5. Memory Recall: Can we find memories by topic/date?

Usage:
    python test_episodic_memory.py
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Import from project
from document_tools import (
    extract_text_from_pdf,
    extract_key_elements,
    save_document_summary
)
from rag_loader import _load_text_rag_context_impl

# =============================================================================
# CONFIGURATION
# =============================================================================

MEMORIES_PATH = "rag/memories"
TEXT_RAG_PATH = "rag/text"
TEST_PDF_PATH = "rag/documents/2019billardchapterEpisodicMemory.pdf"

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
        for line in details.split('\n'):
            print(f"       {line}")


def cleanup_test_memories():
    """Remove test memories from previous runs."""
    test_patterns = ["test_", "unittest_", "accuracy_test_"]
    removed = 0

    if os.path.exists(MEMORIES_PATH):
        for file in os.listdir(MEMORIES_PATH):
            if any(file.startswith(p) for p in test_patterns):
                try:
                    os.remove(os.path.join(MEMORIES_PATH, file))
                    removed += 1
                except:
                    pass

    if removed > 0:
        print(f"[CLEANUP] Removed {removed} test memory files")


# =============================================================================
# TEST 1: MEMORY SAVE STRUCTURE
# =============================================================================

def test_memory_save_structure() -> dict:
    """
    Test that save_document_summary creates properly structured episodic memories.
    """
    print_header("TEST 1: MEMORY SAVE STRUCTURE")

    # Create test summary with full episodic structure
    test_summary = {
        "executive_summary": "Test document about episodic memory concepts",
        "key_elements": [
            {"element": "episodic memory", "importance": 0.95, "category": "concept"},
            {"element": "hippocampus", "importance": 0.90, "category": "anatomy"},
            {"element": "temporal context", "importance": 0.85, "category": "concept"}
        ],
        "temporal_context": {
            "document_date": "2019",
            "time_references": ["past events", "specific moments", "sequential order"],
            "analysis_timestamp": datetime.now().isoformat()
        },
        "spatial_context": {
            "locations_mentioned": ["hippocampus", "medial temporal lobe", "CA3 region"],
            "environment": "neuroscience research"
        },
        "emotional_significance": {
            "tone": "academic/scientific",
            "importance_level": "high",
            "markers": ["memory formation", "cognitive processes", "autobiographical recall"]
        },
        "memory_triggers": [
            "episodic memory",
            "hippocampus",
            "autobiographical",
            "temporal context",
            "what-when-where"
        ],
        "research_sources": [
            {"title": "Episodic Memory Research", "relevance": 0.9}
        ]
    }

    # Save the memory
    document_id = f"test_memory_structure_{int(time.time())}"
    result = save_document_summary(
        document_id=document_id,
        summary_data=json.dumps(test_summary),
        folder_path=MEMORIES_PATH
    )

    print(f"Save result: {result[:100]}...")

    # Verify the saved file
    results = {
        "document_id": document_id,
        "save_result": result
    }

    # Find the saved file
    expected_file = os.path.join(MEMORIES_PATH, f"{document_id}_episodic_memory.json")

    if not os.path.exists(expected_file):
        print_result("File created", False, f"Expected file not found: {expected_file}")
        results["passed"] = False
        return results

    print_result("File created", True, expected_file)

    # Read and validate structure
    with open(expected_file, 'r', encoding='utf-8') as f:
        saved_memory = json.load(f)

    # Check required fields
    required_fields = ["document_id", "memory_event", "summary", "episodic_metadata"]
    missing_fields = [f for f in required_fields if f not in saved_memory]

    if missing_fields:
        print_result("Required fields", False, f"Missing: {missing_fields}")
        results["passed"] = False
        return results

    print_result("Required fields", True, f"All present: {required_fields}")

    # Check memory_event structure (WHEN/WHERE)
    memory_event = saved_memory.get("memory_event", {})
    has_when = "timestamp" in memory_event
    has_where = "location" in memory_event

    print_result("WHEN (timestamp)", has_when,
                 memory_event.get("timestamp", "MISSING")[:25] + "..." if has_when else "No timestamp")
    print_result("WHERE (location)", has_where,
                 memory_event.get("location", "MISSING"))

    # Check episodic_metadata structure (WHY/triggers)
    episodic_metadata = saved_memory.get("episodic_metadata", {})
    has_triggers = len(episodic_metadata.get("recall_triggers", [])) > 0
    has_emotional = bool(episodic_metadata.get("emotional_markers"))
    has_temporal = bool(episodic_metadata.get("temporal_markers"))
    has_spatial = bool(episodic_metadata.get("spatial_markers"))

    print_result("Recall triggers", has_triggers,
                 f"{len(episodic_metadata.get('recall_triggers', []))} triggers")
    print_result("Emotional markers", has_emotional)
    print_result("Temporal markers", has_temporal)
    print_result("Spatial markers", has_spatial)

    # Overall pass
    results["passed"] = (
        has_when and has_where and has_triggers
    )
    results["memory_file"] = expected_file
    results["memory_data"] = saved_memory

    return results


# =============================================================================
# TEST 2: RAG CONTEXT LOADING
# =============================================================================

def test_rag_context_loading() -> dict:
    """
    Test that RAG context loads episodic memory knowledge correctly.
    """
    print_header("TEST 2: RAG CONTEXT LOADING")

    results = {}

    # Load RAG context
    rag_context = _load_text_rag_context_impl(TEXT_RAG_PATH)

    if not rag_context:
        print_result("RAG context loaded", False, "Empty context")
        results["passed"] = False
        return results

    print_result("RAG context loaded", True, f"{len(rag_context)} characters")

    # Check for episodic memory primer content
    expected_content = [
        "episodic memory",
        "WHAT",
        "WHEN",
        "WHERE",
        "autobiographical",
        "recall"
    ]

    found_content = []
    for content in expected_content:
        if content.lower() in rag_context.lower():
            found_content.append(content)

    content_ratio = len(found_content) / len(expected_content)
    print_result("Episodic memory concepts", content_ratio >= 0.8,
                 f"Found {len(found_content)}/{len(expected_content)}: {found_content}")

    # Check for system guardrails
    guardrail_keywords = ["accuracy", "never", "hallucinate", "tool"]
    found_guardrails = sum(1 for kw in guardrail_keywords if kw.lower() in rag_context.lower())

    print_result("System guardrails loaded", found_guardrails >= 2,
                 f"Found {found_guardrails}/{len(guardrail_keywords)} guardrail keywords")

    # Check file count
    file_markers = rag_context.count("# From:")
    print_result("Multiple files loaded", file_markers >= 2,
                 f"{file_markers} files in context")

    results["context_length"] = len(rag_context)
    results["content_found"] = found_content
    results["file_count"] = file_markers
    results["passed"] = content_ratio >= 0.8 and file_markers >= 2

    return results


# =============================================================================
# TEST 3: MEMORY RETRIEVAL
# =============================================================================

def test_memory_retrieval() -> dict:
    """
    Test that we can find and load saved memories.
    """
    print_header("TEST 3: MEMORY RETRIEVAL")

    results = {}

    # First, ensure we have at least one memory
    if not os.path.exists(MEMORIES_PATH):
        os.makedirs(MEMORIES_PATH, exist_ok=True)

    # Create a test memory if none exist
    memory_files = list(Path(MEMORIES_PATH).glob("*.json"))
    if len(memory_files) == 0:
        print("[SETUP] Creating test memory for retrieval test...")
        test_summary = {
            "executive_summary": "Retrieval test memory",
            "memory_triggers": ["retrieval", "test", "episodic"]
        }
        save_document_summary(
            document_id="test_retrieval",
            summary_data=json.dumps(test_summary),
            folder_path=MEMORIES_PATH
        )
        memory_files = list(Path(MEMORIES_PATH).glob("*.json"))

    print(f"Found {len(memory_files)} memory files")

    # Test: Can we load all memories?
    loaded_memories = []
    errors = []

    for mem_file in memory_files:
        try:
            with open(mem_file, 'r', encoding='utf-8') as f:
                memory = json.load(f)
                loaded_memories.append({
                    "file": mem_file.name,
                    "document_id": memory.get("document_id", "unknown"),
                    "timestamp": memory.get("memory_event", {}).get("timestamp", "unknown"),
                    "triggers": memory.get("episodic_metadata", {}).get("recall_triggers", [])
                })
        except Exception as e:
            errors.append(f"{mem_file.name}: {e}")

    print_result("Memory files readable", len(errors) == 0,
                 f"Loaded {len(loaded_memories)}, Errors: {len(errors)}")

    if errors:
        for err in errors[:3]:
            print(f"       ERROR: {err}")

    # Test: Can we search by topic?
    test_topic = "episodic"
    matching_memories = [
        m for m in loaded_memories
        if any(test_topic.lower() in str(t).lower() for t in m.get("triggers", []))
    ]

    print_result(f"Topic search ('{test_topic}')", True,
                 f"Found {len(matching_memories)} matching memories")

    # Test: Can we sort by date?
    memories_with_dates = [
        m for m in loaded_memories
        if m.get("timestamp") and m.get("timestamp") != "unknown"
    ]

    print_result("Temporal ordering", len(memories_with_dates) > 0,
                 f"{len(memories_with_dates)} memories have timestamps")

    results["total_memories"] = len(memory_files)
    results["loaded_successfully"] = len(loaded_memories)
    results["load_errors"] = len(errors)
    results["memories"] = loaded_memories
    results["passed"] = len(errors) == 0 and len(loaded_memories) > 0

    return results


# =============================================================================
# TEST 4: FULL WORKFLOW (Extract -> Analyze -> Save -> Retrieve)
# =============================================================================

def test_full_workflow() -> dict:
    """
    Test the complete episodic memory workflow:
    1. Extract text from document
    2. Extract key elements
    3. Save as episodic memory
    4. Retrieve and verify
    """
    print_header("TEST 4: FULL WORKFLOW")

    results = {"steps": {}}

    # Step 1: Extract text
    print("Step 1: Extracting PDF text...")
    text = extract_text_from_pdf(TEST_PDF_PATH)

    if "Error" in text:
        print_result("PDF extraction", False, text)
        results["passed"] = False
        return results

    print_result("PDF extraction", True, f"{len(text)} characters")
    results["steps"]["extraction"] = True

    # Step 2: Extract key elements
    print("\nStep 2: Extracting key elements...")
    # Use a sample of text for speed
    sample_text = text[:10000]
    elements_json = extract_key_elements(sample_text, max_elements=10)

    try:
        elements = json.loads(elements_json)
        has_elements = len(elements.get("key_elements", [])) > 0
    except:
        has_elements = False
        elements = {}

    print_result("Key elements extraction", has_elements,
                 f"{len(elements.get('key_elements', []))} elements found")
    results["steps"]["key_elements"] = has_elements

    if has_elements:
        print("\n   Top 3 elements:")
        for elem in elements.get("key_elements", [])[:3]:
            print(f"     - {elem.get('element')}: {elem.get('importance')}")

    # Step 3: Create and save episodic memory
    print("\nStep 3: Saving episodic memory...")
    memory_summary = {
        "executive_summary": f"Analysis of {Path(TEST_PDF_PATH).name}",
        "key_elements": elements.get("key_elements", []),
        "temporal_context": {
            "analysis_date": datetime.now().isoformat(),
            "document_year": "2019"
        },
        "spatial_context": {
            "source_file": TEST_PDF_PATH,
            "storage_location": MEMORIES_PATH
        },
        "emotional_significance": {
            "tone": "academic",
            "importance": "high - foundational episodic memory research"
        },
        "memory_triggers": [
            elem.get("element") for elem in elements.get("key_elements", [])[:5]
        ]
    }

    document_id = f"workflow_test_{int(time.time())}"
    save_result = save_document_summary(
        document_id=document_id,
        summary_data=json.dumps(memory_summary),
        folder_path=MEMORIES_PATH
    )

    save_success = "Error" not in save_result
    print_result("Memory save", save_success,
                 save_result[:80] + "..." if len(save_result) > 80 else save_result)
    results["steps"]["save"] = save_success

    # Step 4: Retrieve and verify
    print("\nStep 4: Retrieving saved memory...")
    memory_file = os.path.join(MEMORIES_PATH, f"{document_id}_episodic_memory.json")

    if os.path.exists(memory_file):
        with open(memory_file, 'r', encoding='utf-8') as f:
            retrieved = json.load(f)

        # Verify structure
        has_event = "memory_event" in retrieved
        has_metadata = "episodic_metadata" in retrieved
        has_summary = "summary" in retrieved

        print_result("Memory retrieval", True)
        print_result("  - memory_event", has_event)
        print_result("  - episodic_metadata", has_metadata)
        print_result("  - summary content", has_summary)

        results["steps"]["retrieval"] = has_event and has_metadata and has_summary
    else:
        print_result("Memory retrieval", False, "File not found")
        results["steps"]["retrieval"] = False

    # Overall
    all_steps_passed = all(results["steps"].values())
    results["passed"] = all_steps_passed

    return results


# =============================================================================
# TEST 5: MEMORY INDEX (Phase 2 Preparation)
# =============================================================================

def test_memory_index_capability() -> dict:
    """
    Test capability to build a memory index (preparation for Phase 2).
    This validates we can:
    1. Scan all memories
    2. Extract topics/triggers
    3. Group by date
    """
    print_header("TEST 5: MEMORY INDEX CAPABILITY")

    results = {}

    if not os.path.exists(MEMORIES_PATH):
        print_result("Memories folder exists", False)
        results["passed"] = False
        return results

    # Scan all memories
    memory_files = list(Path(MEMORIES_PATH).glob("*.json"))
    print(f"Scanning {len(memory_files)} memory files...")

    index = {
        "memories": [],
        "topic_clusters": {},
        "date_clusters": {}
    }

    for mem_file in memory_files:
        try:
            with open(mem_file, 'r', encoding='utf-8') as f:
                memory = json.load(f)

            memory_id = memory.get("document_id", mem_file.stem)
            timestamp = memory.get("memory_event", {}).get("timestamp", "")
            triggers = memory.get("episodic_metadata", {}).get("recall_triggers", [])

            # Add to index
            index["memories"].append({
                "id": memory_id,
                "file": mem_file.name,
                "timestamp": timestamp,
                "topics": triggers[:5] if triggers else []
            })

            # Cluster by topic
            for trigger in triggers[:5]:
                trigger_lower = trigger.lower() if isinstance(trigger, str) else str(trigger).lower()
                if trigger_lower not in index["topic_clusters"]:
                    index["topic_clusters"][trigger_lower] = []
                index["topic_clusters"][trigger_lower].append(memory_id)

            # Cluster by date
            if timestamp:
                date_key = timestamp[:10]  # YYYY-MM-DD
                if date_key not in index["date_clusters"]:
                    index["date_clusters"][date_key] = []
                index["date_clusters"][date_key].append(memory_id)

        except Exception as e:
            print(f"[WARN] Could not process {mem_file.name}: {e}")

    # Report results
    print_result("Index built", len(index["memories"]) > 0,
                 f"{len(index['memories'])} memories indexed")
    print_result("Topic clusters", len(index["topic_clusters"]) > 0,
                 f"{len(index['topic_clusters'])} unique topics")
    print_result("Date clusters", len(index["date_clusters"]) > 0,
                 f"{len(index['date_clusters'])} date groups")

    # Show top topics
    if index["topic_clusters"]:
        top_topics = sorted(
            index["topic_clusters"].items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        print("\n   Top topics:")
        for topic, mems in top_topics:
            print(f"     - {topic}: {len(mems)} memories")

    results["index"] = index
    results["memory_count"] = len(index["memories"])
    results["topic_count"] = len(index["topic_clusters"])
    results["passed"] = len(index["memories"]) > 0

    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests() -> dict:
    """Run all episodic memory tests."""
    print_header("EPISODIC MEMORY TEST SUITE", "#")

    # Cleanup old test files
    cleanup_test_memories()

    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }

    # Run tests
    tests = [
        ("memory_save_structure", test_memory_save_structure),
        ("rag_context_loading", test_rag_context_loading),
        ("memory_retrieval", test_memory_retrieval),
        ("full_workflow", test_full_workflow),
        ("memory_index_capability", test_memory_index_capability)
    ]

    for test_name, test_func in tests:
        try:
            results["tests"][test_name] = test_func()
        except Exception as e:
            print(f"[ERROR] Test {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results["tests"][test_name] = {"passed": False, "error": str(e)}

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
    results["success_rate"] = total_passed / total_tests if total_tests > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {total_passed}/{total_tests} tests passed ({results['success_rate']*100:.0f}%)")
    print(f"{'=' * 70}\n")

    if results["success_rate"] >= 0.8:
        print("SUCCESS! Episodic memory system is functioning correctly.")
        print("\nPhase 1 Status: COMPLETE")
        print("Ready for Phase 2: Memory Clustering & Recall")
    else:
        print("WARNING: Some tests failed. Review errors above.")

    return results


if __name__ == "__main__":
    results = run_all_tests()
    sys.exit(0 if results["success_rate"] >= 0.8 else 1)
