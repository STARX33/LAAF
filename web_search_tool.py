"""
Web search tool for research and fact-checking using DuckDuckGo.
Designed to enhance document intelligence with external knowledge.
"""
from typing import List, Dict
from smolagents import tool
import time

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    print("[WARNING] duckduckgo-search not installed. Web search disabled.")
    print("Install with: pip install duckduckgo-search")


@tool
def search_web_for_research(query: str, max_results: int = 5) -> str:
    """
    Search the web for relevant research sources and information.

    Args:
        query: Search query (can be a question or keywords)
        max_results: Maximum number of results to return (default: 5, max: 10)

    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    if not HAS_DDGS:
        return "Error: Web search not available. Install duckduckgo-search library."

    # Limit max_results to prevent overwhelming output
    max_results = min(max_results, 10)

    # Retry logic for rate limiting
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return f"No search results found for: {query}"

            break  # Success, exit retry loop

        except Exception as retry_error:
            if "ratelimit" in str(retry_error).lower() or "202" in str(retry_error):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    return f"Web search rate limited after {max_retries} attempts. Please try again later."
            else:
                # Different error, raise it
                raise

    try:

        # Format results
        output = [f"Web Search Results for: '{query}'"]
        output.append("=" * 70)

        for idx, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('href', result.get('link', 'No URL'))
            snippet = result.get('body', result.get('description', 'No description'))

            output.append(f"\n{idx}. {title}")
            output.append(f"   URL: {url}")
            output.append(f"   {snippet[:200]}..." if len(snippet) > 200 else f"   {snippet}")

        output.append("\n" + "=" * 70)
        return "\n".join(output)

    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def find_research_sources(topic: str, num_sources: int = 5) -> str:
    """
    Find 3-5 relevant research sources for a given topic.
    Optimized for academic and technical topics.

    Args:
        topic: The research topic or document subject
        num_sources: Number of sources to find (default: 5)

    Returns:
        Formatted list of research sources with relevance scores
    """
    if not HAS_DDGS:
        return "Error: Web search not available. Install duckduckgo-search library."

    # Enhance query for better research results
    research_query = f"{topic} research academic papers documentation"

    # Retry logic for rate limiting
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(research_query, max_results=num_sources * 2))

            if not results:
                return f"No research sources found for: {topic}"

            break  # Success, exit retry loop

        except Exception as retry_error:
            if "ratelimit" in str(retry_error).lower() or "202" in str(retry_error):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    return f"Web search rate limited after {max_retries} attempts. Please try again later."
            else:
                raise

    try:

        # Filter and score results
        scored_results = []
        for result in results:
            title = result.get('title', '')
            url = result.get('href', result.get('link', ''))
            snippet = result.get('body', result.get('description', ''))

            # Calculate simple relevance score
            relevance = 0.5  # Base score

            # Boost for academic indicators
            academic_keywords = ['research', 'study', 'paper', 'journal', 'academic', 'documentation', 'technical']
            for keyword in academic_keywords:
                if keyword.lower() in title.lower() or keyword.lower() in snippet.lower():
                    relevance += 0.05

            # Boost for credible domains
            credible_domains = ['.edu', '.gov', '.org', 'github.com', 'arxiv.org', 'ieee.org']
            if any(domain in url.lower() for domain in credible_domains):
                relevance += 0.15

            relevance = min(relevance, 1.0)  # Cap at 1.0

            scored_results.append({
                'title': title,
                'url': url,
                'snippet': snippet,
                'relevance': relevance
            })

        # Sort by relevance and take top N
        scored_results.sort(key=lambda x: x['relevance'], reverse=True)
        top_results = scored_results[:num_sources]

        # Format output
        output = [f"Research Sources for: '{topic}'"]
        output.append("=" * 70)
        output.append(f"Found {len(top_results)} relevant sources (sorted by relevance)\n")

        for idx, source in enumerate(top_results, 1):
            output.append(f"{idx}. {source['title']}")
            output.append(f"   URL: {source['url']}")
            output.append(f"   Relevance: {source['relevance']:.2f}")
            output.append(f"   {source['snippet'][:150]}..." if len(source['snippet']) > 150 else f"   {source['snippet']}")
            output.append("")

        output.append("=" * 70)
        return "\n".join(output)

    except Exception as e:
        return f"Error finding research sources: {str(e)}"
