import streamlit as st
import streamlit.components.v1 as components
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from collections import Counter
from datetime import datetime
import time

# PWA Support
def add_pwa_support():
    manifest = {"name": "Insight Ink", "short_name": "Insight Ink", "start_url": "/", "display": "standalone", "background_color": "#ffffff", "theme_color": "#1f77b4", "icons": [{"src": "icon-192.png", "sizes": "192x192", "type": "image/png"}, {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"}]}
    import json
    m = json.dumps(manifest)
    html = f'<meta name="theme-color" content="#1f77b4"><meta name="mobile-web-app-capable" content="yes"><script>const b=new Blob(["{m}"],{{type:"application/json"}});const u=URL.createObjectURL(b);const l=document.createElement("link");l.rel="manifest";l.href=u;document.head.appendChild(l);if("serviceWorker" in navigator){{const sw=`self.addEventListener("install",e=>self.skipWaiting());self.addEventListener("activate",e=>self.clients.claim());`;const sb=new Blob([sw],{{type:"application/javascript"}});const su=URL.createObjectURL(sb);navigator.serviceWorker.register(su).catch(()=>{{}});}}</script>'
    components.html(html, height=0)

# Auto-refresh script
def add_auto_refresh(interval=30):
    """Auto-refresh the page every X seconds"""
    html = f"""
    <script>
        setTimeout(function(){{
            window.parent.location.reload();
        }}, {interval * 1000});
    </script>
    """
    components.html(html, height=0)

# NLTK Setup
for c in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{c}' if c == 'stopwords' else f'tokenizers/{c}')
    except:
        nltk.download(c, quiet=True)

# Categories with LIVE news sources
CATEGORIES = {
    "🔴 Breaking News": [
        ("BBC Breaking", "https://www.bbc.com/news"),
        ("Reuters Breaking", "https://www.reuters.com/"),
        ("CNN Breaking", "https://edition.cnn.com/")
    ],
    "🌍 World": [
        ("BBC World", "https://www.bbc.com/news/world"),
        ("Reuters World", "https://www.reuters.com/world/"),
        ("CNN World", "https://edition.cnn.com/world"),
        ("Al Jazeera", "https://www.aljazeera.com/")
    ],
    "💼 Business": [
        ("BBC Business", "https://www.bbc.com/news/business"),
        ("Reuters Business", "https://www.reuters.com/business/"),
        ("Bloomberg", "https://www.bloomberg.com/"),
        ("Financial Times", "https://www.ft.com/")
    ],
    "💻 Technology": [
        ("BBC Tech", "https://www.bbc.com/news/technology"),
        ("Reuters Tech", "https://www.reuters.com/technology/"),
        ("TechCrunch", "https://techcrunch.com/"),
        ("The Verge", "https://www.theverge.com/")
    ],
    "🏥 Health": [
        ("BBC Health", "https://www.bbc.com/news/health"),
        ("Reuters Health", "https://www.reuters.com/business/healthcare-pharmaceuticals/"),
        ("WebMD News", "https://www.webmd.com/news/")
    ],
    "⚽ Sports": [
        ("BBC Sport", "https://www.bbc.com/sport"),
        ("ESPN", "https://www.espn.com/"),
        ("Reuters Sports", "https://www.reuters.com/sports/"),
        ("Sky Sports", "https://www.skysports.com/")
    ],
    "🎬 Entertainment": [
        ("BBC Entertainment", "https://www.bbc.com/news/entertainment_and_arts"),
        ("Variety", "https://variety.com/"),
        ("Hollywood Reporter", "https://www.hollywoodreporter.com/"),
        ("Deadline", "https://deadline.com/")
    ],
    "🇮🇳 India": [
        ("The Hindu", "https://www.thehindu.com/news/"),
        ("NDTV", "https://www.ndtv.com/"),
        ("Times of India", "https://timesofindia.indiatimes.com/"),
        ("Indian Express", "https://indianexpress.com/")
    ]
}

# Clean title
def clean_title(title, content):
    t = title.strip()
    for s in ['BBC', 'Reuters', 'CNN', 'Hindu', 'NDTV', 'ESPN', 'Variety', 'TechCrunch', 'Bloomberg']:
        t = t.replace(f' - {s}', '').replace(f'| {s}', '').replace(f': {s}', '')
    t = t.replace('NewsNews', '').replace('Latest', '').replace('Breaking', '')
    t = ' '.join(t.split())
    
    if not t or len(t) < 10:
        sents = sent_tokenize(content)
        if sents:
            t = sents[0].split(',')[0] if ',' in sents[0] else sents[0]
    
    if len(t) > 70:
        t = t.split(' - ')[0] if ' - ' in t else ' '.join(t.split()[:12]) + '...'
    
    if t.isupper():
        t = t.title()
    return t.strip(' .-:') or "Article"

# Get keywords
def get_keywords(text, n=4):
    try:
        t = text.lower()
        words = word_tokenize(t)
        stops = set(stopwords.words('english'))
        stops.update(['said', 'new', 'year', 'people', 'time', 'day', 'also', 'would', 'could', 'report', 'news'])
        filtered = [w for w in words if len(w) >= 4 and w.isalpha() and w not in stops]
        if not filtered:
            return []
        from nltk.util import ngrams
        bg = [' '.join(b) for b in ngrams(filtered, 2)]
        wf = Counter(filtered)
        bf = Counter(bg)
        cands = [(w.capitalize(), f*2) for w, f in wf.most_common(n*2) if f >= 2]
        cands += [(' '.join([x.capitalize() for x in b.split()]), f*3) for b, f in bf.most_common(n) if f >= 2]
        seen = set()
        unique = []
        for k, s in cands:
            if k.lower() not in seen:
                seen.add(k.lower())
                unique.append((k, s))
        unique.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in unique[:n]]
    except:
        return []

# Fetch article
def fetch_article(url):
    h = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', 'Accept': 'text/html'}
    try:
        r = requests.get(url, headers=h, timeout=10)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        for t in soup(["script", "style", "nav", "footer", "header"]):
            t.decompose()
        title = soup.find('h1')
        title = title.get_text(strip=True) if title else "Article"
        text = ""
        arts = soup.find_all(['article', 'div'], class_=lambda x: x and any(k in str(x).lower() for k in ['article', 'story', 'content', 'post']))
        if arts:
            for a in arts:
                ps = a.find_all('p')
                text += ' '.join([p.get_text(strip=True) for p in ps if p.get_text(strip=True)])
                if len(text) > 200:
                    break
        if len(text) < 200:
            ps = soup.find_all('p')
            text = ' '.join([p.get_text(strip=True) for p in ps if len(p.get_text(strip=True)) > 50])
        text = ' '.join(text.split())
        if text and len(text) > 100:
            return {'title': title, 'url': url, 'content': text, 'time': datetime.now()}
    except:
        pass
    return None

# Fetch category with pagination support
def fetch_category(url, max_articles=20):
    h = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    articles = []
    try:
        r = requests.get(url, headers=h, timeout=15)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            if any(p in href.lower() for p in ['/article', '/story', '/news', '/world', '/tech', '/business', '/sport', '/entertainment']):
                if any(s in href.lower() for s in ['video', 'gallery', 'podcast', 'live-reporting']):
                    continue
                if href.startswith('http'):
                    links.add(href)
                elif href.startswith('/'):
                    links.add('/'.join(url.split('/')[:3]) + href)
            if len(links) >= max_articles * 3:
                break
        
        fetched = 0
        for link in list(links):
            if fetched >= max_articles:
                break
            a = fetch_article(link)
            if a and a['url'] not in [art['url'] for art in articles]:
                articles.append(a)
                fetched += 1
    except:
        pass
    return articles

# Summarize
def summarize(text, n=3):
    sents = sent_tokenize(text)
    if len(sents) <= n:
        return ' '.join(sents)
    try:
        vec = TfidfVectorizer(stop_words='english')
        sv = vec.fit_transform(sents)
        dv = vec.transform([' '.join(sents)])
        scores = cosine_similarity(sv, dv).flatten()
        top = scores.argsort()[-n:][::-1]
        return ' '.join([sents[i] for i in sorted(top.tolist())])
    except:
        return ' '.join(sents[:n])

# Time ago formatter
def time_ago(dt):
    now = datetime.now()
    diff = now - dt
    seconds = diff.total_seconds()
    if seconds < 60:
        return "Just now"
    elif seconds < 3600:
        return f"{int(seconds/60)}m ago"
    elif seconds < 86400:
        return f"{int(seconds/3600)}h ago"
    else:
        return f"{int(seconds/86400)}d ago"

# Streamlit App
st.set_page_config(layout="wide", page_title="Insight Ink LIVE", page_icon="🔴")
add_pwa_support()

st.markdown("""
<style>
.live-badge{background:linear-gradient(90deg,#ff0000,#ff4444);color:white;padding:4px 10px;border-radius:15px;font-size:0.8rem;font-weight:bold;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.7}}
.keyword-tag{display:inline-block;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:5px 12px;border-radius:20px;margin:3px;font-size:0.85rem;font-weight:600}
.article-card{background:#f8f9fa;border-radius:10px;padding:1.5rem;margin:1rem 0;border-left:4px solid #1f77b4;box-shadow:0 2px 4px rgba(0,0,0,0.1)}
.time-badge{background:#28a745;color:white;padding:3px 8px;border-radius:10px;font-size:0.75rem;font-weight:bold}
.new-badge{background:#ff6b6b;color:white;padding:3px 8px;border-radius:10px;font-size:0.75rem;font-weight:bold;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.5}}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'category' not in st.session_state:
    st.session_state.category = None
if 'source' not in st.session_state:
    st.session_state.source = None
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'articles_per_load' not in st.session_state:
    st.session_state.articles_per_load = 10
if 'total_loaded' not in st.session_state:
    st.session_state.total_loaded = 0

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🔴 Insight Ink LIVE")
    st.markdown("*Real-time News Streaming*")
with col2:
    if st.session_state.category and st.session_state.source:
        st.markdown('<span class="live-badge">● LIVE</span>', unsafe_allow_html=True)

st.info("💡 **Mobile:** Tap menu (⋮) → 'Add to Home screen'")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Live Controls")
    
    if st.session_state.category and st.session_state.source:
        st.success(f"📡 Streaming: {st.session_state.source[0]}")
        
        auto_refresh = st.checkbox("🔄 Auto-Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        if auto_refresh:
            refresh_interval = st.slider("Refresh every (seconds)", 15, 120, 30)
            add_auto_refresh(refresh_interval)
            st.info(f"🔄 Auto-refreshing every {refresh_interval}s")
        
        articles_to_load = st.selectbox("Articles per load", [5, 10, 15, 20], index=1)
        st.session_state.articles_per_load = articles_to_load
        
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        if st.button("📥 Load More", use_container_width=True):
            with st.spinner("Loading more articles..."):
                new_articles = fetch_category(st.session_state.source[1], st.session_state.articles_per_load)
                for art in new_articles:
                    if art['url'] not in [a['url'] for a in st.session_state.articles]:
                        st.session_state.articles.append(art)
            st.rerun()
        
        st.markdown("---")
        if st.button("🏠 Back to Categories", use_container_width=True):
            st.session_state.category = None
            st.session_state.source = None
            st.session_state.articles = []
            st.rerun()
    
    st.markdown("---")
    st.subheader("📊 Stats")
    st.metric("Articles Loaded", len(st.session_state.articles))
    if st.session_state.articles:
        st.metric("Latest Update", time_ago(st.session_state.articles[0]['time']))

# Main content
if not st.session_state.category:
    st.header("📚 Select News Category")
    cols = st.columns(2)
    cats = list(CATEGORIES.keys())
    for i, cat in enumerate(cats):
        with cols[i % 2]:
            if st.button(cat, key=f"c{i}", use_container_width=True):
                st.session_state.category = cat
                st.session_state.articles = []
                st.rerun()

elif not st.session_state.source:
    st.subheader(f"{st.session_state.category} - Choose Source")
    sources = CATEGORIES[st.session_state.category]
    scols = st.columns(min(len(sources), 4))
    for i, (name, url) in enumerate(sources):
        with scols[i % 4]:
            if st.button(f"📰 {name}", key=f"s{i}", use_container_width=True):
                st.session_state.source = (name, url)
                with st.spinner(f"Loading live news from {name}..."):
                    st.session_state.articles = fetch_category(url, st.session_state.articles_per_load)
                st.rerun()

else:
    # Display articles
    if st.session_state.articles:
        st.success(f"📰 {len(st.session_state.articles)} articles loaded from {st.session_state.source[0]}")
        
        for i, art in enumerate(st.session_state.articles):
            title = clean_title(art['title'], art['content'])
            summary = summarize(art['content'], 3)
            keywords = get_keywords(art['content'], 4)
            
            # Article card
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"### 📌 {title}")
            with col2:
                if i < 3:
                    st.markdown('<span class="new-badge">NEW</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="time-badge">{time_ago(art["time"])}</span>', unsafe_allow_html=True)
            
            if keywords:
                kw_html = ''.join([f'<span class="keyword-tag">{k}</span>' for k in keywords])
                st.markdown(kw_html, unsafe_allow_html=True)
            
            st.markdown("**✨ Quick Summary:**")
            st.write(summary)
            st.markdown(f"🔗 [Read Full Article]({art['url']})")
            st.markdown("---")
        
        # Load more button at bottom
        if st.button("📥 Load More Articles", key="load_bottom", use_container_width=True):
            with st.spinner("Fetching more news..."):
                new_articles = fetch_category(st.session_state.source[1], st.session_state.articles_per_load)
                for art in new_articles:
                    if art['url'] not in [a['url'] for a in st.session_state.articles]:
                        st.session_state.articles.append(art)
            st.rerun()
    
    else:
        st.warning("No articles found. Try refreshing or selecting a different source.")
        if st.button("🔄 Try Again"):
            with st.spinner("Loading..."):
                st.session_state.articles = fetch_category(st.session_state.source[1], st.session_state.articles_per_load)
            st.rerun()

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray'><p>Made with ❤️ | Real-time news updates</p></div>", unsafe_allow_html=True)