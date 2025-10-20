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

# PWA Support
def add_pwa_support():
    manifest = {"name": "Insight Ink", "short_name": "Insight Ink", "start_url": "/", "display": "standalone", "background_color": "#ffffff", "theme_color": "#1f77b4", "icons": [{"src": "icon-192.png", "sizes": "192x192", "type": "image/png"}, {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"}]}
    import json
    m = json.dumps(manifest)
    html = f'<meta name="theme-color" content="#1f77b4"><meta name="mobile-web-app-capable" content="yes"><script>const b=new Blob(["{m}"],{{type:"application/json"}});const u=URL.createObjectURL(b);const l=document.createElement("link");l.rel="manifest";l.href=u;document.head.appendChild(l);if("serviceWorker" in navigator){{const sw=`self.addEventListener("install",e=>self.skipWaiting());self.addEventListener("activate",e=>self.clients.claim());`;const sb=new Blob([sw],{{type:"application/javascript"}});const su=URL.createObjectURL(sb);navigator.serviceWorker.register(su).catch(()=>{{}});}}</script>'
    components.html(html, height=0)

# NLTK Setup
for c in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{c}' if c == 'stopwords' else f'tokenizers/{c}')
    except:
        nltk.download(c, quiet=True)

# Categories
CATEGORIES = {
    "🌍 World": [("BBC", "https://www.bbc.com/news/world"), ("Reuters", "https://www.reuters.com/world/"), ("CNN", "https://edition.cnn.com/world")],
    "💼 Business": [("BBC", "https://www.bbc.com/news/business"), ("Reuters", "https://www.reuters.com/business/")],
    "💻 Technology": [("BBC", "https://www.bbc.com/news/technology"), ("Reuters", "https://www.reuters.com/technology/")],
    "🏥 Health": [("BBC", "https://www.bbc.com/news/health"), ("Reuters", "https://www.reuters.com/business/healthcare-pharmaceuticals/")],
    "⚽ Sports": [("BBC", "https://www.bbc.com/sport"), ("ESPN", "https://www.espn.com/")],
    "🎬 Entertainment": [("BBC", "https://www.bbc.com/news/entertainment_and_arts"), ("Variety", "https://variety.com/")],
    "🇮🇳 India": [("The Hindu", "https://www.thehindu.com/news/"), ("NDTV", "https://www.ndtv.com/")]
}

MAX_ARTICLES = 5

# Clean title
def clean_title(title, content):
    t = title.strip()
    for s in ['BBC', 'Reuters', 'CNN', 'Hindu', 'NDTV', 'ESPN', 'Variety']:
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
    h = {'User-Agent': 'Mozilla/5.0', 'Accept': 'text/html'}
    try:
        r = requests.get(url, headers=h, timeout=15)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        for t in soup(["script", "style", "nav", "footer", "header"]):
            t.decompose()
        title = soup.find('h1')
        title = title.get_text(strip=True) if title else "Article"
        text = ""
        arts = soup.find_all(['article', 'div'], class_=lambda x: x and any(k in str(x).lower() for k in ['article', 'story', 'content']))
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
            return {'title': title, 'url': url, 'content': text}
    except:
        pass
    return None

# Fetch category
def fetch_category(url):
    h = {'User-Agent': 'Mozilla/5.0'}
    articles = []
    try:
        r = requests.get(url, headers=h, timeout=15)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            if any(p in href.lower() for p in ['/article', '/story', '/news', '/world', '/tech', '/business']):
                if any(s in href.lower() for s in ['video', 'gallery', 'podcast']):
                    continue
                if href.startswith('http'):
                    links.add(href)
                elif href.startswith('/'):
                    links.add('/'.join(url.split('/')[:3]) + href)
            if len(links) >= MAX_ARTICLES * 3:
                break
        for link in list(links)[:MAX_ARTICLES * 2]:
            if len(articles) >= MAX_ARTICLES:
                break
            a = fetch_article(link)
            if a:
                articles.append(a)
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

# Streamlit App
st.set_page_config(layout="wide", page_title="Insight Ink", page_icon="📰")
add_pwa_support()

st.markdown("""
<style>
.keyword-tag{display:inline-block;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:5px 12px;border-radius:20px;margin:3px;font-size:0.85rem;font-weight:600}
</style>
""", unsafe_allow_html=True)

st.title("📰 Insight Ink")
st.markdown("*Your Smart News Companion*")
st.info("💡 **Mobile:** Tap menu (⋮) → 'Add to Home screen'")
st.markdown("---")

if 'category' not in st.session_state:
    st.session_state.category = None
if 'articles' not in st.session_state:
    st.session_state.articles = []

st.header("📚 Choose Category")

cols = st.columns(2)
cats = list(CATEGORIES.keys())

for i, cat in enumerate(cats):
    with cols[i % 2]:
        if st.button(cat, key=f"c{i}", use_container_width=True):
            st.session_state.category = cat
            st.session_state.articles = []
            st.rerun()

if st.session_state.category:
    st.markdown("---")
    st.subheader(st.session_state.category)
    sources = CATEGORIES[st.session_state.category]
    scols = st.columns(len(sources))
    for i, (name, url) in enumerate(sources):
        with scols[i]:
            if st.button(f"📰 {name}", key=f"s{i}", use_container_width=True):
                with st.spinner(f"Fetching {name}..."):
                    st.session_state.articles = fetch_category(url)
                st.rerun()

if st.session_state.articles:
    st.markdown("---")
    st.success(f"✅ {len(st.session_state.articles)} articles")
    for art in st.session_state.articles:
        title = clean_title(art['title'], art['content'])
        summary = summarize(art['content'], 3)
        keywords = get_keywords(art['content'], 4)
        
        st.markdown(f"### 📌 {title}")
        if keywords:
            kw_html = ''.join([f'<span class="keyword-tag">{k}</span>' for k in keywords])
            st.markdown(kw_html, unsafe_allow_html=True)
        st.markdown("**✨ Summary:**")
        st.write(summary)
        st.markdown(f"🔗 [Read Full]({art['url']})")
        st.markdown("---")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("Browse news by category with AI summaries!")
    st.markdown("---")
    st.subheader("Features")
    st.write("✓ 7 Categories")
    st.write("✓ Smart Titles")
    st.write("✓ AI Summaries")
    st.write("✓ Key Topics")
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.category = None
        st.session_state.articles = []
        st.rerun()

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray'><p>Made with ❤️</p></div>", unsafe_allow_html=True)