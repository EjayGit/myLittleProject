#!/usr/bin/env python3
"""
ZIM Reader with LLM Text Cleaning
"""

from flask import Flask, request, jsonify
from libzim.reader import Archive
from libzim.search import Query, Searcher
import os
import re
from bs4 import BeautifulSoup

app = Flask(__name__)

# ZIM file
ZIM_PATH = r"E:\WikiFull\wikipedia_en_all_nopic_2025-12.zim"
zim = Archive(ZIM_PATH)

print(f"✅ ZIM loaded: {zim.entry_count:,} entries")

def clean_text_for_llm(text):
    """
    Clean text for LLM training - removes newlines and extra spaces
    """
    if not text:
        return text
    
    # Replace all newline variants with space
    text = text.replace('\r\n', ' ')  # Windows
    text = text.replace('\n', ' ')     # Unix
    text = text.replace('\r', ' ')     # Old Mac
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Trim
    text = text.strip()
    
    return text

def extract_article_content(entry, clean_for_llm=False):
    """
    Extract text from article entry
    
    Args:
        entry: ZIM entry object
        clean_for_llm: If True, clean text for LLM training
    """
    try:
        if not hasattr(entry, 'get_item'):
            return "No get_item method"
        
        item = entry.get_item()
        content = item.content
        
        # Convert memoryview to bytes if needed
        if isinstance(content, memoryview):
            content = content.tobytes()
        
        if isinstance(content, bytes):
            # Decode the content
            try:
                content_str = content.decode('utf-8', errors='ignore')
            except:
                try:
                    content_str = content.decode('latin-1', errors='ignore')
                except:
                    return f"Could not decode content ({len(content)} bytes)"
            
            # Check if it looks like HTML
            content_start = content_str[:500].lower()
            is_html = ('<!doctype' in content_start or 
                      '<html' in content_start or 
                      'http-equiv' in content_start or
                      '<body' in content_start)
            
            if is_html:
                # Use BeautifulSoup for HTML parsing
                try:
                    soup = BeautifulSoup(content_str, 'html.parser')
                    
                    # Remove unwanted elements
                    for elem in soup(['script', 'style', 'nav', 'footer', 'table', 
                                    'sup', 'span', 'div.navbox', 'div.vertical-navbox',
                                    'div.reflist', 'ol.references', 'div.citation',
                                    'span.mw-editsection', 'span.mw-headline']):
                        elem.decompose()
                    
                    # Find and remove the "See also" section
                    see_also_h2 = soup.find('h2', {'id': 'See_also'})
                    
                    if see_also_h2:
                        # Remove everything from this h2 onward
                        for elem in see_also_h2.find_all_next():
                            elem.decompose()
                        see_also_h2.decompose()  # Remove the h2 itself
                    
                    # Try to find main content
                    content_div = soup.find('div', {'id': 'mw-content-text'})
                    if content_div:
                        text = content_div.get_text(separator=' ', strip=True) if clean_for_llm else content_div.get_text(separator='\n', strip=True)
                    else:
                        text = soup.get_text(separator=' ', strip=True) if clean_for_llm else soup.get_text(separator='\n', strip=True)
                    
                    # Clean up
                    if clean_for_llm:
                        # Already cleaned by BeautifulSoup with space separator
                        text = clean_text_for_llm(text)
                    
                    return text
                    
                except Exception as e:
                    print(f"   BeautifulSoup failed: {e}")
                    # Fallback: simple regex (but no "see also" check needed here)
                    text = re.sub(r'<[^>]+>', ' ', content_str)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if clean_for_llm:
                        text = clean_text_for_llm(text)
                    
                    return text
            
            else:
                # Not HTML - no "see also" checking needed
                text = content_str
                
                if clean_for_llm:
                    text = clean_text_for_llm(text)
                
                return text
        
        else:
            return f"Unexpected content type: {type(content)}"
            
    except Exception as e:
        return f"Error extracting content: {str(e)}"

@app.route('/')
def home():
    """Serve the HTML page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ZIM Reader - LLM Training Prep</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 15px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .control-panel { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; border: 1px solid #dee2e6; }
            .btn { padding: 12px 24px; font-size: 16px; border: none; border-radius: 6px; cursor: pointer; margin: 5px; }
            .btn-primary { background: #3498db; color: white; }
            .btn-primary:hover { background: #2980b9; }
            .btn-success { background: #2ecc71; color: white; }
            .btn-success:hover { background: #27ae60; }
            .btn-warning { background: #f39c12; color: white; }
            .btn-warning:hover { background: #e67e22; }
            input[type="text"] { padding: 12px; width: 400px; font-size: 16px; border: 1px solid #ddd; border-radius: 6px; }
            .status { padding: 15px; margin: 15px 0; border-radius: 6px; font-weight: bold; }
            .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .result-item { 
                padding: 15px; 
                border: 1px solid #dee2e6; 
                margin: 10px 0; 
                cursor: pointer;
                background: white;
                border-radius: 6px;
                transition: all 0.2s;
            }
            .result-item:hover { 
                background: #e9ecef; 
                border-color: #adb5bd;
                transform: translateY(-2px);
            }
            .article-content { 
                white-space: pre-wrap; 
                line-height: 1.6; 
                font-size: 16px;
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 6px;
                border: 1px solid #dee2e6;
                max-height: 600px;
                overflow-y: auto;
            }
            .llm-text { 
                white-space: normal; 
                line-height: 1.8; 
                font-size: 16px;
                text-align: justify;
            }
            .stats { 
                background: #e8f4f8; 
                padding: 15px; 
                border-radius: 6px; 
                margin: 15px 0;
                border-left: 4px solid #3498db;
            }
            .toggle { display: flex; align-items: center; margin: 10px 0; }
            .toggle input { margin-right: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 Wikipedia ZIM Reader - LLM Training Prep</h1>
            
            <div class="control-panel">
                <h3>🔍 Search Articles</h3>
                <input type="text" id="search" placeholder="Search (e.g., quantum, science, computer)" value="artificial intelligence">
                <button class="btn btn-primary" onclick="searchArticles()">Search</button>
                
                <div class="toggle">
                    <input type="checkbox" id="llmMode" checked>
                    <label for="llmMode"><strong>LLM Mode:</strong> Clean text (no newlines, single paragraphs)</label>
                </div>
                
                <div class="stats">
                    <strong>📊 ZIM Stats:</strong><br>
                    <span id="stats">Loading...</span>
                </div>
            </div>
            
            <div id="status" class="status info">Ready to search...</div>
            
            <div id="results">
                <div class="result-item" style="background: #f0f0f0; cursor: default; font-style: italic;">
                    Search results will appear here. Try: "science", "history", "mathematics", "computer science"
                </div>
            </div>
            
            <div id="article">
                <h3>📖 Article Content</h3>
                <p>Select an article to view its content here.</p>
                <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 6px; border: 1px solid #ffeaa7;">
                    <strong>💡 LLM Training Tip:</strong> For best results, enable "LLM Mode" to get clean, continuous text without newlines or extra spaces.
                </div>
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: #e8f5e9; border-radius: 6px; border: 1px solid #c8e6c9;">
                <h3>🚀 LLM Training Export</h3>
                <p>Ready to train your model? Use the API endpoints below:</p>
                <div style="background: white; padding: 15px; border-radius: 6px; font-family: monospace; margin: 10px 0;">
                    # Get clean text for LLM training<br>
                    GET /article?path=A/Anarchism&clean_for_llm=true<br><br>
                    # Search and get all results for batch processing<br>
                    GET /search?q=science&limit=100<br><br>
                    # Batch extract multiple articles<br>
                    GET /batch_extract?q=science&limit=10&clean_for_llm=true
                </div>
                <button class="btn btn-success" onclick="showExportHelp()">Show Export Examples</button>
            </div>
        </div>

        <script>
            async function searchArticles() {
                const query = document.getElementById('search').value.trim();
                if (!query) {
                    showStatus('Please enter a search term', 'error');
                    return;
                }
                
                showStatus(`Searching for "${query}"...`, 'info');
                
                try {
                    const response = await fetch(`/search?q=${encodeURIComponent(query)}&limit=20`);
                    const data = await response.json();
                    
                    if (data.error) {
                        showStatus(`Error: ${data.error}`, 'error');
                        return;
                    }
                    
                    let html = `<h3>🔍 Search Results (${data.total} found)</h3>`;
                    
                    if (data.results.length === 0) {
                        html += '<p>No results found. Try a different search term.</p>';
                    } else {
                        data.results.forEach((result, index) => {
                            const safePath = result.path.replace(/'/g, "\\'").replace(/"/g, '&quot;');
                            html += `
                                <div class="result-item" onclick="loadArticle('${safePath}')">
                                    <strong>${escapeHtml(result.title)}</strong><br>
                                    <small>${escapeHtml(result.path)}</small>
                                </div>
                            `;
                        });
                    }
                    
                    document.getElementById('results').innerHTML = html;
                    showStatus(`✅ Found ${data.total} results for "${query}"`, 'success');
                    
                } catch (error) {
                    showStatus(`❌ Search failed: ${error.message}`, 'error');
                }
            }
            
            async function loadArticle(path) {
                const llmMode = document.getElementById('llmMode').checked;
                showStatus('Loading article...', 'info');
                
                try {
                    const url = `/article?path=${encodeURIComponent(path)}&clean_for_llm=${llmMode}`;
                    const response = await fetch(url);
                    const data = await response.json();
                    
                    if (data.error) {
                        showStatus(`❌ Error: ${data.error}`, 'error');
                        return;
                    }
                    
                    const className = llmMode ? 'llm-text' : 'article-content';
                    
                    document.getElementById('article').innerHTML = `
                        <h2>${escapeHtml(data.title)}</h2>
                        <div class="stats">
                            <strong>📊 Article Info:</strong><br>
                            • Path: ${escapeHtml(data.path)}<br>
                            • Length: ${data.length.toLocaleString()} characters<br>
                            • Mode: ${data.clean_for_llm ? 'LLM Cleaned' : 'Original Format'}
                        </div>
                        <hr>
                        <div class="${className}">${escapeHtml(data.text)}</div>
                        
                        <div style="margin-top: 20px;">
                            <button class="btn btn-warning" onclick="copyToClipboard()">📋 Copy Text</button>
                            <button class="btn btn-primary" onclick="saveAsFile()">💾 Save as .txt</button>
                        </div>
                    `;
                    
                    showStatus(`✅ Article loaded (${data.length.toLocaleString()} chars)`, 'success');
                    
                } catch (error) {
                    showStatus(`❌ Failed to load article: ${error.message}`, 'error');
                }
            }
            
            async function loadStats() {
                try {
                    const response = await fetch('/stats');
                    const data = await response.json();
                    
                    document.getElementById('stats').innerHTML = `
                        • Entries: ${data.entries.toLocaleString()}<br>
                        • Articles: ${data.articles.toLocaleString()}<br>
                        • Search: ${data.has_search ? '✅ Available' : '❌ Not available'}
                    `;
                } catch (error) {
                    document.getElementById('stats').innerHTML = 'Error loading stats';
                }
            }
            
            function copyToClipboard() {
                const text = document.querySelector('.llm-text, .article-content')?.textContent;
                if (text) {
                    navigator.clipboard.writeText(text)
                        .then(() => showStatus('✅ Text copied to clipboard', 'success'))
                        .catch(() => showStatus('❌ Failed to copy', 'error'));
                }
            }
            
            function saveAsFile() {
                const text = document.querySelector('.llm-text, .article-content')?.textContent;
                const title = document.querySelector('h2')?.textContent || 'article';
                
                if (text) {
                    const blob = new Blob([text], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.txt`;
                    a.click();
                    URL.revokeObjectURL(url);
                    showStatus('✅ File saved', 'success');
                }
            }
            
            function showExportHelp() {
                alert(`📋 Export Examples:\n\n` +
                      `1. Direct API call for clean text:\n` +
                      `   http://127.0.0.1:5001/article?path=A/Anarchism&clean_for_llm=true\n\n` +
                      `2. Batch processing script:\n` +
                      `   python batch_extract.py --query "science" --limit 100 --output science_articles.json\n\n` +
                      `3. For your Mamba training:\n` +
                      `   Use the /batch_extract endpoint to get training data in clean format.`);
            }
            
            function showStatus(message, type) {
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = message;
                statusDiv.className = 'status ' + type;
            }
            
            function escapeHtml(text) {
                if (!text) return '';
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            // Initialize
            window.onload = function() {
                loadStats();
                
                // Auto-search after 1 second
                setTimeout(() => {
                    if (document.getElementById('search').value) {
                        searchArticles();
                    }
                }, 1000);
                
                // Enter key support
                document.getElementById('search').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        searchArticles();
                    }
                });
            };
        </script>
    </body>
    </html>
    '''

@app.route('/test')
def test():
    return jsonify({
        'status': 'ok',
        'zim_loaded': True,
        'entries': zim.entry_count,
        'has_search': zim.has_fulltext_index
    })

@app.route('/search')
def search():
    """Search API"""
    query = request.args.get('q', '').strip()
    limit = request.args.get('limit', 20, type=int)
    
    print(f"\n🔍 Search: '{query}' (limit: {limit})")
    
    if not query:
        return jsonify({'error': 'No query'}), 400
    
    try:
        if not zim.has_fulltext_index:
            return jsonify({
                'error': 'This ZIM file does not have a search index',
                'results': [],
                'total': 0
            })
        
        # Perform search
        search_query = Query().set_query(query)
        searcher = Searcher(zim)
        results = searcher.search(search_query)
        
        count = results.getEstimatedMatches()
        print(f"   Found {count} results")
        
        # Get results
        result_paths = results.getResults(0, min(limit, count))
        
        # Format results
        formatted = []
        for path in result_paths:
            try:
                entry = zim.get_entry_by_path(path)
                formatted.append({
                    'title': entry.title,
                    'path': path,
                    'url': getattr(entry, 'url', path)
                })
            except Exception as e:
                formatted.append({
                    'title': f'Error: {str(e)[:50]}',
                    'path': path,
                    'url': path
                })
        
        return jsonify({
            'query': query,
            'total': count,
            'results': formatted
        })
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/article')
def article():
    """Get article API with LLM cleaning option"""
    path = request.args.get('path', '').strip()
    clean_for_llm = request.args.get('clean_for_llm', 'false').lower() == 'true'
    
    print(f"\n📖 Article request: '{path}'")
    print(f"   LLM Cleaning: {clean_for_llm}")
    
    if not path:
        return jsonify({'error': 'No path'}), 400
    
    try:
        if not zim.has_entry_by_path(path):
            return jsonify({'error': f'Article not found: {path}'}), 404
        
        entry = zim.get_entry_by_path(path)
        
        # Get content with LLM cleaning option
        text = ''
        if hasattr(entry, 'get_item'):
            try:
                text = extract_article_content(entry, clean_for_llm=clean_for_llm)
                print(f"   Extracted {len(text):,} characters")
                
                # Apply final LLM cleaning if requested
                if clean_for_llm:
                    text = clean_text_for_llm(text)
                    print(f"   Final cleaned length: {len(text):,} characters")
                
            except Exception as e:
                print(f"   Error extracting content: {e}")
                text = f"Error extracting content: {str(e)}"
        else:
            text = "Entry has no get_item method"
        
        return jsonify({
            'title': entry.title,
            'path': path,
            'url': getattr(entry, 'url', path),
            'text': text,
            'length': len(text),
            'clean_for_llm': clean_for_llm
        })
        
    except Exception as e:
        print(f"❌ Article error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_extract')
def batch_extract():
    """Batch extract articles for LLM training"""
    query = request.args.get('q', '').strip()
    limit = request.args.get('limit', 10, type=int)
    clean_for_llm = request.args.get('clean_for_llm', 'true').lower() == 'true'
    
    print(f"\n📚 Batch extract: '{query}' (limit: {limit}, clean: {clean_for_llm})")
    
    if not query:
        return jsonify({'error': 'No query'}), 400
    
    try:
        if not zim.has_fulltext_index:
            return jsonify({'error': 'No search index available'}), 400
        
        # Perform search
        search_query = Query().set_query(query)
        searcher = Searcher(zim)
        results = searcher.search(search_query)
        
        count = results.getEstimatedMatches()
        result_paths = results.getResults(0, min(limit, count))
        
        # Extract articles
        articles = []
        for path in result_paths:
            try:
                entry = zim.get_entry_by_path(path)
                text = extract_article_content(entry, clean_for_llm=clean_for_llm)
                
                if clean_for_llm:
                    text = clean_text_for_llm(text)
                
                if text and len(text) > 100:  # Minimum length
                    articles.append({
                        'title': entry.title,
                        'path': path,
                        'text': text,
                        'length': len(text)
                    })
                    
            except Exception as e:
                print(f"   Skipping {path}: {e}")
                continue
        
        print(f"   Extracted {len(articles)} articles")
        
        return jsonify({
            'query': query,
            'total_found': count,
            'extracted': len(articles),
            'articles': articles,
            'clean_for_llm': clean_for_llm
        })
        
    except Exception as e:
        print(f"❌ Batch extract error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Statistics API"""
    return jsonify({
        'entries': zim.entry_count,
        'articles': zim.article_count,
        'has_search': zim.has_fulltext_index
    })

if __name__ == '__main__':
    print(f"🌐 Server starting: http://127.0.0.1:5001")
    print(f"📊 Total entries: {zim.entry_count:,}")
    print(f"🔍 Search index: {'✅ Available' if zim.has_fulltext_index else '❌ Not available'}")
    print("\n🚀 LLM Training Endpoints:")
    print("  /article?path=A/Anarchism&clean_for_llm=true")
    print("  /batch_extract?q=science&limit=10&clean_for_llm=true")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=True, host='127.0.0.1', port=5001)
