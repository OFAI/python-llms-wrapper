# Web Search Integration for LLMs

## Overview
This document summarizes approaches and tools for integrating web search capabilities with LLMs, specifically for use with the SerialChatbot class in the llms_wrapper package.

## Search APIs and Services

### Official Search Engine APIs
- **Google Custom Search JSON API**: Official Google API for programmatic search
  - Requires API key, has usage limits
  - Generally reliable but has costs
- **Bing Web Search API**: Microsoft's search API
  - Part of Azure Cognitive Services
  - Commercial service with usage-based pricing
- **DuckDuckGo Instant Answer API**: Privacy-focused search
  - No API key required, but limited functionality
  - Good for simple queries

### Self-Hosted Solutions
- **Searx/SearxNG**: Self-hosted meta-search engine
  - Can be used as API endpoint
  - Privacy-focused, no external dependencies
  - Requires server setup and maintenance

### Commercial Search API Aggregators
- **SerpAPI**: Commercial search API aggregator
  - Paid service, supports multiple search engines
  - Handles rate limiting and blocking issues
  - More reliable than scraping approaches

## Python Libraries for Web Search

### Important Note
Many GitHub repositories and package names change over time. Always verify current availability by:
1. Searching PyPI directly for current packages
2. Checking GitHub's search for "python web search" or "python duckduckgo"
3. Looking at current LangChain documentation for their web search integrations

### General Approaches
- **DuckDuckGo scrapers**: Various Python libraries exist that scrape DuckDuckGo results
- **Google scrapers**: Libraries like `googlesearch-python` (unofficial, may be blocked)
- **Search API wrappers**: Libraries that wrap official APIs

## Web Content Extraction

### HTML to Markdown Conversion
- **`html2text`**: Convert HTML to markdown
  - Good for basic HTML content
  - Simple to use and integrate
- **`markdownify`**: Another HTML to markdown converter
  - More customizable than html2text
  - Better handling of complex HTML structures
- **`trafilatura`**: Web content extraction library
  - Excellent for extracting main content from web pages
  - Handles boilerplate removal automatically

### JavaScript-Rendered Content
- **`selenium`**: Web browser automation
  - Full browser automation, handles JS rendering
  - More resource-intensive but comprehensive
- **`playwright`**: Modern browser automation
  - Faster than Selenium, better for scraping
  - Good Python support and documentation
- **`requests-html`**: JavaScript support for requests
  - Simpler than Selenium for basic JS rendering
  - Good middle ground for simple JS sites

### Headless Browsers
- **`pyppeteer`**: Python port of Puppeteer
  - Chrome/Chromium automation
  - Good for complex JavaScript interactions
- **`splash`**: Lightweight browser for scraping
  - Docker-based, good for complex JS sites
  - Can be deployed as a service

## Integrated Solutions

### LangChain Tools
- **`langchain-community`** includes web search tools:
  - Various search API wrappers
  - Web content loaders
  - Pre-built integration patterns

### Specialized Libraries
- **`newspaper3k`**: Article extraction and curation
  - Good for news articles and blogs
  - Handles common article formats well
- **`readability-lxml`**: Extract readable content
  - Based on Mozilla's readability algorithm
  - Good for cleaning up web content

## Implementation Patterns

### 1. Simple Search + Scraping
```python
# General pattern with any search library + content extraction
def search_and_extract(query, num_results=5):
    # Search using chosen library
    results = search_library.search(query, max_results=num_results)
    
    # Extract content from each result
    extracted_content = []
    for result in results:
        downloaded = fetch_url(result['url'])
        content = extract_content(downloaded, output_format='markdown')
        extracted_content.append({
            'url': result['url'],
            'title': result['title'],
            'content': content
        })
    
    return extracted_content
```

### 2. JavaScript-Enabled Search
```python
# Pattern for JS-heavy sites
def search_with_js(query):
    with browser_automation_tool() as browser:
        page = browser.new_page()
        page.goto(f"https://searchengine.com/?q={query}")
        
        # Wait for results to load
        page.wait_for_selector('.result')
        
        # Extract content
        content = page.content()
        
        # Convert to markdown
        return convert_to_markdown(content)
```

### 3. Integration with SerialChatbot
```python
class WebSearchTool:
    def __init__(self, search_engine='duckduckgo'):
        self.search_engine = search_engine
    
    def search_and_summarize(self, query, max_results=3):
        # Implement search + content extraction
        # Return markdown-formatted results
        pass
    
    def get_page_content(self, url):
        # Extract content from specific URL
        # Return as markdown
        pass

# Add to SerialChatbot
class SerialChatbot:
    def __init__(self, ...):
        # ... existing code ...
        self.web_search = WebSearchTool()
    
    def reply_with_search(self, message, search_query=None):
        # Perform search, add results to context
        # Then generate response
        pass
```

## Recommendations

### For General Use
1. **DuckDuckGo scraper** + **content extraction library** - No API keys needed
2. **Commercial API** + **content extraction library** - If you need reliability and don't mind paying

### For JavaScript-Heavy Sites
1. **Playwright** + **HTML to markdown converter** - Modern, fast, reliable
2. **Selenium** + **content extraction library** - More established, wider browser support

### For Production Systems
1. **Self-hosted Searx** + **content extraction** - Privacy-focused, no external dependencies
2. **Commercial APIs** - Reliable, compliant, but with costs

### Integration Patterns
- **Tool-based**: Add search as a tool that LLM can call when needed
- **Context-enhanced**: Pre-search relevant information and add to context
- **Retrieval-augmented**: Use search results as external knowledge base
- **Interactive**: Let users trigger searches through chat commands

## Considerations

### Technical
- **Rate limiting**: Many services have rate limits
- **Blocking**: Scrapers may be blocked by search engines
- **JavaScript**: Many modern sites require JS rendering
- **Content quality**: Need to filter and clean extracted content

### Legal and Ethical
- **Terms of service**: Check ToS for scraping restrictions
- **API compliance**: Follow API usage guidelines
- **Privacy**: Consider user privacy implications
- **Caching**: Implement appropriate caching to reduce requests

### Performance
- **Latency**: Web requests add significant latency
- **Resource usage**: Browser automation is resource-intensive
- **Caching**: Cache results to improve performance
- **Async processing**: Consider async operations for better UX

## Next Steps

1. Research current available packages on PyPI
2. Test different approaches with sample queries
3. Implement basic search functionality
4. Add content extraction and markdown conversion
5. Integrate with SerialChatbot class
6. Add error handling and rate limiting
7. Consider caching strategies
8. Implement user interface for search commands