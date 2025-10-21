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
import io
import os
import importlib.util
import subprocess
import sys

# --- Auto-install helper for missing packages (best-effort) ---
def try_install(package_name, import_name=None):
    """Try to import a package; if missing attempt pip install (best-effort)."""
    target = import_name or package_name
    if importlib.util.find_spec(target) is None:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        except Exception as e:
            try:
                st.warning(f"Could not auto-install {package_name}: {e}")
            except:
                pass

# Try to ensure deep_translator and gtts are available (best-effort)
try_install("deep-translator", "deep_translator")
try_install("gTTS", "gtts")

# Import deep_translator if available
try:
    from deep_translator import GoogleTranslator
except Exception as e:
    GoogleTranslator = None
    try:
        st.warning(f"deep_translator import failed: {e}")
    except:
        pass

# Import gTTS for server-side audio generation
try:
    from gtts import gTTS
except Exception as e:
    gTTS = None
    try:
        st.warning(f"gTTS import failed: {e}")
    except:
        pass
# ----------------------------------------------------------------

# PWA Support
def add_pwa_support():
    manifest = {
        "name": "Insight Ink Pro",
        "short_name": "Insight Ink",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0a0e27",
        "theme_color": "#6366f1",
        "icons": [
            {"src": "icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"}
        ]
    }
    import json
    # ensure manifest json won't break injected script
    m = json.dumps(manifest).replace('"', '\\"')
    html = f"""
    <meta name="theme-color" content="#6366f1">
    <meta name="mobile-web-app-capable" content="yes">
    <script>
    try {{
        const manifestData = "{m}";
        const b = new Blob([manifestData], {{type:"application/json"}});
        const u = URL.createObjectURL(b);
        const l = document.createElement("link");
        l.rel = "manifest";
        l.href = u;
        document.head.appendChild(l);
        if("serviceWorker" in navigator){{
            const sw=`self.addEventListener("install",e=>self.skipWaiting());self.addEventListener("activate",e=>self.clients.claim());`;
            const sb=new Blob([sw],{{type:"application/javascript"}});
            const su=URL.createObjectURL(sb);
            navigator.serviceWorker.register(su).catch(()=>{{}});
        }}
    }} catch(e) {{
        console.warn('PWA manifest injection error', e);
    }}
    </script>
    """
    components.html(html, height=0)

def add_auto_refresh(interval=30):
    html = f'<script>setTimeout(function(){{window.parent.location.reload();}},{interval*1000});</script>'
    components.html(html, height=0)

# NLTK Setup
for c in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        # corpora vs tokenizers path detection
        if c == 'stopwords':
            nltk.data.find(f'corpora/{c}')
        else:
            nltk.data.find(f'tokenizers/{c}')
    except:
        try:
            nltk.download(c, quiet=True)
        except:
            pass

# Languages
LANGUAGES = {
    "🇬🇧 English": ("en", "en-US"),
    "🇮🇳 हिंदी": ("hi", "hi-IN"),
    "🇮🇳 मराठी": ("mr", "mr-IN"),
    "🇮🇳 ಕನ್ನಡ": ("kn", "kn-IN")
}

# Translation dictionaries for UI
TRANSLATIONS = {
    "hi": {
        "Quick Summary": "त्वरित सारांश",
        "Key Topics": "मुख्य विषय",
        "Read Full": "पूरा पढ़ें",
        "Listen": "सुनें",
        "Just now": "अभी",
        "articles loaded": "लेख लोड",
        "Loading": "लोड हो रहा है",
        "Load 10 More": "10 और लोड करें"
    },
    "mr": {
        "Quick Summary": "द्रुत सारांश",
        "Key Topics": "मुख्य विषय",
        "Read Full": "संपूर्ण वाचा",
        "Listen": "ऐका",
        "Just now": "आत्ताच",
        "articles loaded": "लेख लोड",
        "Loading": "लोड होत आहे",
        "Load 10 More": "आणखी 10 लोड करा"
    },
    "kn": {
        "Quick Summary": "ತ್ವರಿತ ಸಾರಾಂಶ",
        "Key Topics": "ಪ್ರಮುಖ ವಿಷಯಗಳು",
        "Read Full": "ಸಂಪೂರ್ಣ ಓದಿ",
        "Listen": "ಕೇಳಿ",
        "Just now": "ಈಗ",
        "articles loaded": "ಲೇಖನಗಳು",
        "Loading": "ಲೋಡ್ ಆಗುತ್ತಿದೆ",
        "Load 10 More": "ಇನ್ನೂ 10 ಲೋಡ್"
    }
}

def translate_ui(text, lang_code):
    if lang_code == "en":
        return text
    return TRANSLATIONS.get(lang_code, {}).get(text, text)

# Translation of summary text using deep_translator (chunking)
@st.cache_data(ttl=3600)
def translate_text(text, target_lang):
    if GoogleTranslator is None:
        return text
    if not text or target_lang == "en":
        return text
    try:
        max_len = 4500
        if len(text) <= max_len:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
        # chunk by sentences
        sents = sent_tokenize(text)
        out = []
        cur = ""
        for s in sents:
            if len(cur) + len(s) + 1 < max_len:
                cur += s + " "
            else:
                out.append(GoogleTranslator(source='auto', target=target_lang).translate(cur.strip()))
                cur = s + " "
        if cur.strip():
            out.append(GoogleTranslator(source='auto', target=target_lang).translate(cur.strip()))
        return " ".join(out)
    except Exception as e:
        try:
            st.warning(f"Translation error: {e}")
        except:
            pass
        return text

# Categories (same as original)
CATEGORIES = {
    "🔴 Breaking": [
        ("BBC", "https://www.bbc.com/news"),
        ("Reuters", "https://www.reuters.com/"),
        ("CNN", "https://edition.cnn.com/"),
        ("Al Jazeera", "https://www.aljazeera.com/")
    ],
    "🌍 World": [
        ("BBC World", "https://www.bbc.com/news/world"),
        ("Reuters World", "https://www.reuters.com/world/"),
        ("CNN World", "https://edition.cnn.com/world"),
        ("Guardian", "https://www.theguardian.com/world")
    ],
    "💼 Business": [
        ("BBC Business", "https://www.bbc.com/news/business"),
        ("Reuters Business", "https://www.reuters.com/business/"),
        ("Bloomberg", "https://www.bloomberg.com/"),
        ("CNBC", "https://www.cnbc.com/")
    ],
    "💻 Tech": [
        ("BBC Tech", "https://www.bbc.com/news/technology"),
        ("Reuters Tech", "https://www.reuters.com/technology/"),
        ("TechCrunch", "https://techcrunch.com/"),
        ("Verge", "https://www.theverge.com/")
    ],
    "🏥 Health": [
        ("BBC Health", "https://www.bbc.com/news/health"),
        ("Reuters Health", "https://www.reuters.com/business/healthcare-pharmaceuticals/"),
        ("WebMD", "https://www.webmd.com/news/"),
        ("MedNews", "https://www.medicalnewstoday.com/")
    ],
    "⚽ Sports": [
        ("BBC Sport", "https://www.bbc.com/sport"),
        ("ESPN", "https://www.espn.com/"),
        ("Reuters Sports", "https://www.reuters.com/sports/"),
        ("Sky Sports", "https://www.skysports.com/")
    ],
    "🎬 Entertainment": [
        ("BBC Arts", "https://www.bbc.com/news/entertainment_and_arts"),
        ("Variety", "https://variety.com/"),
        ("Hollywood", "https://www.hollywoodreporter.com/"),
        ("Deadline", "https://deadline.com/")
    ],
    "🇮🇳 India": [
        ("Hindu", "https://www.thehindu.com/news/"),
        ("NDTV", "https://www.ndtv.com/"),
        ("TOI", "https://timesofindia.indiatimes.com/"),
        ("Express", "https://indianexpress.com/")
    ]
}

# Title cleaning and robust extraction
def extract_title_from_soup(soup):
    """
    Try multiple metadata locations to extract a reliable title:
      1) og:title
      2) meta name="title"
      3) <title> tag (page title)
      4) h1
    Return the raw title string (minimal cleaning).
    """
    # 1) og:title
    og = soup.find('meta', property='og:title')
    if og and og.get('content'):
        return og['content'].strip()
    # 2) meta name=title
    meta_title = soup.find('meta', attrs={'name':'title'})
    if meta_title and meta_title.get('content'):
        return meta_title['content'].strip()
    # 3) <title>
    ttag = soup.find('title')
    if ttag and ttag.get_text(strip=True):
        return ttag.get_text(strip=True)
    # 4) h1
    h1 = soup.find('h1')
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    return "Article"

def clean_title_for_display(raw_title):
    """
    Light touch cleaning for display: remove repeated site suffixes such as ' - BBC' etc
    while preserving the main title text as much as possible.
    Avoid over-normalization so user's requested 'title exactly as fetched' is respected.
    """
    t = raw_title.strip()
    # Remove obvious duplicates like 'NewsNews' or repeated word
    t = re.sub(r'NewsNews', 'News', t, flags=re.IGNORECASE)
    # Remove trailing site names in common formats
    site_tokens = [' - BBC', ' - Reuters', ' - CNN', ' - The Hindu', ' - NDTV', ' - ESPN', ' - Variety', ' - TechCrunch', ' - Bloomberg', ' - Guardian', ' | BBC', ' | Reuters']
    for s in site_tokens:
        if s in t:
            t = t.replace(s, '')
    # Remove trailing separators repeated
    t = re.sub(r'(\s*[-|:]\s*)+$', '', t).strip()
    # If title is empty after cleaning, fallback to original raw_title
    if not t:
        return raw_title.strip()
    return t

# Keyword extraction
def get_keywords(text, n=5):
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
        cands = [(w.capitalize(), f*2) for w, f in wf.most_common(n*3) if f >= 2]
        cands += [(' '.join([x.capitalize() for x in b.split()]), f*4) for b, f in bf.most_common(n*2) if f >= 2]
        seen = set()
        unique = []
        for k, s in cands:
            kl = k.lower()
            if kl not in seen and not any(kl in u.lower() or u.lower() in kl for u, _ in unique):
                seen.add(kl)
                unique.append((k, s))
        unique.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in unique[:n]]
    except:
        return []

# Article fetching logic (improved title extraction)
def fetch_article(url):
    h = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        r = requests.get(url, headers=h, timeout=12)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        # remove boilerplate tags
        for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
            t.decompose()

        title = extract_title_from_soup(soup)  # robust extraction

        text = ""
        # try to find main article containers with keywords in class
        arts = soup.find_all(['article', 'div', 'main'], class_=lambda x: x and any(k in str(x).lower() for k in ['article', 'story', 'content', 'post', 'body']))
        if arts:
            for a in arts[:3]:
                ps = a.find_all('p')
                for p in ps:
                    pt = p.get_text(strip=True)
                    # avoid tiny caption-like paragraphs
                    if len(pt) > 30:
                        text += pt + ' '
                if len(text) > 300:
                    break
        # fallback to all paragraphs
        if len(text) < 200:
            ps = soup.find_all('p')
            for p in ps:
                pt = p.get_text(strip=True)
                if len(pt) > 30:
                    text += pt + ' '
                if len(text) > 500:
                    break
        text = ' '.join(text.split())
        if text and len(text) > 150:
            return {'title': title, 'url': url, 'content': text, 'time': datetime.now()}
    except Exception as e:
        # swallow network errors quietly
        # optional: log or collect errors
        pass
    return None

def fetch_category(url, max_articles=30, offset=0):
    h = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    articles = []
    try:
        r = requests.get(url, headers=h, timeout=20)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if any(p in href.lower() for p in ['/article', '/story', '/news', '/20', '/blog', '/post']):
                if any(s in href.lower() for s in ['video', 'gallery', 'podcast', 'live-reporting', 'javascript:', '#']):
                    continue
                if href.startswith('http'):
                    links.append(href)
                elif href.startswith('/'):
                    base = '/'.join(url.split('/')[:3])
                    links.append(base + href)
            if len(links) >= (max_articles + offset) * 4:
                break

        links = list(dict.fromkeys(links))
        links = links[offset:offset + max_articles * 3]

        fetched = 0
        for link in links:
            if fetched >= max_articles:
                break
            a = fetch_article(link)
            if a:
                articles.append(a)
                fetched += 1
    except:
        pass
    return articles

# Summarization function (same algorithm improved)
def summarize(text, n=6):
    sents = sent_tokenize(text)
    if len(sents) <= n:
        return ' '.join(sents)
    try:
        vec = TfidfVectorizer(stop_words='english', max_features=200, ngram_range=(1,2))
        sv = vec.fit_transform(sents)
        dv = vec.transform([' '.join(sents)])
        scores = cosine_similarity(sv, dv).flatten()

        if len(scores) > 0:
            scores[0] *= 1.3

        top = scores.argsort()[-n:][::-1]
        summary_sents = [sents[i] for i in sorted(top.tolist())]

        if 0 not in top and len(sents[0].split()) < 30:
            summary_sents = [sents[0]] + summary_sents[1:]

        return ' '.join(summary_sents)
    except:
        return ' '.join(sents[:n])

def time_ago(dt):
    diff = datetime.now() - dt
    s = diff.total_seconds()
    if s < 60:
        return "Just now"
    elif s < 3600:
        return f"{int(s/60)}m"
    elif s < 86400:
        return f"{int(s/3600)}h"
    else:
        return f"{int(s/86400)}d"

# ---------------- Streamlit App UI & Logic ----------------
st.set_page_config(layout="wide", page_title="Insight Ink Pro", page_icon="🚀", initial_sidebar_state="expanded")
add_pwa_support()

# Advanced CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* {font-family: 'Inter', sans-serif !important;}
.main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); color: #e2e8f0; padding-top: 0 !important;}
.stApp {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);}
h1, h2, h3 {color: #f1f5f9 !important;}
.live-badge {background: linear-gradient(90deg, #ef4444, #dc2626); color: white; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 700; animation: pulse 2s infinite; display: inline-block; box-shadow: 0 0 20px rgba(239,68,68,0.5);}
@keyframes pulse {0%, 100% {opacity: 1; transform: scale(1);} 50% {opacity: 0.8; transform: scale(0.98);}}
.article-card {background: linear-gradient(135deg, #1e293b 0%, #2d3748 100%); border-radius: 16px; padding: 2rem; margin: 1.5rem 0; border-left: 5px solid #6366f1; box-shadow: 0 8px 24px rgba(0,0,0,0.4); transition: all 0.3s ease;}
.article-card:hover {transform: translateX(8px); box-shadow: 0 12px 32px rgba(99,102,241,0.3);}
.keyword-tag {display: inline-block; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; padding: 6px 16px; border-radius: 20px; margin: 4px; font-size: 0.8rem; font-weight: 600; box-shadow: 0 4px 12px rgba(99,102,241,0.3); transition: all 0.2s;}
.keyword-tag:hover {transform: translateY(-2px); box-shadow: 0 6px 16px rgba(99,102,241,0.5);}
.time-badge {background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; box-shadow: 0 2px 8px rgba(16,185,129,0.3);}
.new-badge {background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; animation: blink 1.5s infinite; box-shadow: 0 2px 8px rgba(239,68,68,0.4);}
@keyframes blink {0%, 100% {opacity: 1;} 50% {opacity: 0.6;}}
.stat-card {background: linear-gradient(135deg, #2d3748 0%, #1e293b 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid #475569; box-shadow: 0 4px 12px rgba(0,0,0,0.3);}
.stat-number {font-size: 2.5rem; font-weight: 800; color: #6366f1; text-shadow: 0 0 20px rgba(99,102,241,0.5);}
.stat-label {color: #94a3b8; font-size: 0.9rem; font-weight: 600;}
.stButton>button {background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 1.5rem !important; font-weight: 700 !important; transition: all 0.3s !important; box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;}
.stButton>button:hover {transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(99,102,241,0.6) !important;}
.voice-btn {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: white;
    padding: 10px 20px;
    border-radius: 12px;
    font-size: 0.9rem;
    font-weight: 700;
    cursor: pointer;
    display: inline-block;
    margin: 10px 0;
    box-shadow: 0 4px 12px rgba(245,158,11,0.4);
    transition: all 0.3s;
    border: none;
}
.voice-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(245,158,11,0.6);
}
.voice-btn:active {
    transform: translateY(0);
}
[data-testid="stSidebarNav"] {display: none;}
header[data-testid="stHeader"] {display: none !important;}
.sidebar .sidebar-content {background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;}
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
if 'offset' not in st.session_state:
    st.session_state.offset = 0
if 'language' not in st.session_state:
    st.session_state.language = "en"
if 'prev_language' not in st.session_state:
    st.session_state.prev_language = "en"
if 'translated_articles' not in st.session_state:
    st.session_state.translated_articles = {}
if 'audio_cache' not in st.session_state:
    st.session_state.audio_cache = {}  # key -> bytes

# Header
col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    st.markdown("<h1 style='margin:0; padding:0;'>🚀 Insight Ink <span style='color:#6366f1;'>PRO</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; margin:0;'>Real-time AI-Powered News Intelligence</p>", unsafe_allow_html=True)
with col2:
    if st.session_state.category and st.session_state.source:
        st.markdown('<div class="live-badge">● LIVE</div>', unsafe_allow_html=True)
with col3:
    lang_keys = list(LANGUAGES.keys())
    current_lang = [k for k, v in LANGUAGES.items() if v[0] == st.session_state.language]
    lang_index = lang_keys.index(current_lang[0]) if current_lang else 0
    selected_lang = st.selectbox("", lang_keys, index=lang_index, label_visibility="collapsed", key="lang_select")
    new_lang = LANGUAGES[selected_lang][0]

    # Detect language change and rerun
    if new_lang != st.session_state.prev_language:
        st.session_state.language = new_lang
        st.session_state.prev_language = new_lang
        st.session_state.translated_articles = {}  # Clear translation cache
        st.session_state.audio_cache = {}  # Clear audio cache (language changed)
        st.rerun()

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Control Center")

    if st.session_state.category and st.session_state.source:
        st.success(f"📡 {st.session_state.source[0]}")

        st.markdown("#### 🔄 Auto-Refresh")
        auto_refresh = st.toggle("Enable", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh

        if auto_refresh:
            interval = st.slider("Interval (sec)", 20, 120, 40)
            add_auto_refresh(interval)
            st.info(f"⏱️ {interval}s")

        st.markdown("---")

        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.offset = 0
            st.session_state.articles = []
            st.session_state.translated_articles = {}
            st.session_state.audio_cache = {}
            st.rerun()

        if st.button(f"📥 {translate_ui('Load 10 More', st.session_state.language)}", use_container_width=True):
            with st.spinner(translate_ui("Loading", st.session_state.language)):
                st.session_state.offset += 10
                new = fetch_category(st.session_state.source[1], 10, st.session_state.offset)
                for a in new:
                    if a['url'] not in [x['url'] for x in st.session_state.articles]:
                        st.session_state.articles.append(a)
            st.rerun()

        st.markdown("---")

        if st.button("🏠 Home", use_container_width=True):
            st.session_state.category = None
            st.session_state.source = None
            st.session_state.articles = []
            st.session_state.offset = 0
            st.session_state.translated_articles = {}
            st.session_state.audio_cache = {}
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Statistics")
    st.markdown(f'<div class="stat-card"><div class="stat-number">{len(st.session_state.articles)}</div><div class="stat-label">{translate_ui("articles loaded", st.session_state.language)}</div></div>', unsafe_allow_html=True)
    if st.session_state.articles:
        st.markdown(f'<div class="stat-card" style="margin-top:1rem;"><div class="stat-number" style="font-size:1.5rem;">{time_ago(st.session_state.articles[0]["time"])}</div><div class="stat-label">Last Update</div></div>', unsafe_allow_html=True)

# Main content
if not st.session_state.category:
    st.markdown("### 📚 Choose Your Category")
    cols = st.columns(2)
    cats = list(CATEGORIES.keys())
    for i, cat in enumerate(cats):
        with cols[i % 2]:
            if st.button(cat, key=f"c{i}", use_container_width=True):
                st.session_state.category = cat
                st.session_state.articles = []
                st.session_state.offset = 0
                st.session_state.translated_articles = {}
                st.session_state.audio_cache = {}
                st.rerun()

elif not st.session_state.source:
    st.markdown(f"### {st.session_state.category}")
    st.markdown("#### Select News Source")
    sources = CATEGORIES[st.session_state.category]
    cols = st.columns(4)
    for i, (name, url) in enumerate(sources):
        with cols[i % 4]:
            if st.button(f"📰 {name}", key=f"s{i}", use_container_width=True):
                st.session_state.source = (name, url)
                with st.spinner(f"{translate_ui('Loading', st.session_state.language)} {name}..."):
                    st.session_state.articles = fetch_category(url, 15, 0)
                    st.session_state.translated_articles = {}
                    st.session_state.audio_cache = {}
                st.rerun()

else:
    if st.session_state.articles:
        st.success(f"✅ {len(st.session_state.articles)} {translate_ui('articles loaded', st.session_state.language)}")

        for i, art in enumerate(st.session_state.articles):
            # Keep raw fetched title intact for display and speech (as requested)
            fetched_title = art.get('title', 'Article')  # from fetch_article (already uses og/title/h1)
            display_title = clean_title_for_display(fetched_title)

            # Summarize only the content, not the title
            summary_en = summarize(art['content'], 6)

            # Translation cache keyed by article url + target language
            cache_key = f"{art['url']}_{st.session_state.language}_summary"

            if st.session_state.language != "en":
                if cache_key not in st.session_state.translated_articles:
                    with st.spinner(f"🌐 Translating article {i+1}..."):
                        trans = translate_text(summary_en, st.session_state.language)
                        st.session_state.translated_articles[cache_key] = {'summary': trans}
                        summary_trans = trans
                else:
                    summary_trans = st.session_state.translated_articles[cache_key]['summary']
            else:
                summary_trans = summary_en

            keywords = get_keywords(art['content'], 5)

            # Layout: Title and time
            col1, col2 = st.columns([5, 1])
            with col1:
                # Show the original fetched title (unchanged)
                st.markdown(f"### 📌 {display_title}")
            with col2:
                if i < 5:
                    st.markdown('<span class="new-badge">NEW</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="time-badge">{time_ago(art["time"])}</span>', unsafe_allow_html=True)

            # Keywords
            if keywords:
                kw_html = ''.join([f'<span class="keyword-tag">{k}</span>' for k in keywords])
                st.markdown(kw_html, unsafe_allow_html=True)

            # Highlighted summary
            st.markdown(f"**✨ {translate_ui('Quick Summary', st.session_state.language)}:**")
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
                    border-left: 4px solid #6366f1;
                    padding: 14px;
                    border-radius: 8px;
                    color: #f8fafc;
                    line-height:1.6;
                    font-size:1rem;
                    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
                    margin-top:6px;
                    ">
                    {summary_trans}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Voice: generate and play on-click using gTTS (female voice default)
            # We'll generate audio bytes and cache them in session_state.audio_cache
            # Use the fetched_title (unchanged) + summary_trans for the narration
            # But the user asked: "it should summarize the idea of news but not title" — so we will speak the
            # title (as fetched) followed by the summary (translated if requested).
            # Prepare strings and safe language codes for gTTS
            lang_code = st.session_state.language  # 'en', 'hi', 'mr', 'kn'
            # Map to gTTS supported languages if necessary; gTTS supports 'en', 'hi', 'mr' (Marathi support may be limited),
            # 'kn' might not be supported by gTTS; if unsupported, fallback to 'en' or use original language code.
            gtts_supported = {'en', 'hi', 'mr', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'ru', 'ja', 'zh-cn'}
            gtts_lang = lang_code if lang_code in gtts_supported else lang_code.split('-')[0] if lang_code.split('-')[0] in gtts_supported else 'en'

            # Prepare text to speak (title + summary). Title kept as fetched from source.
            to_speak = f"{fetched_title}. {summary_trans}"

            # Ensure to_speak is a string and not unexpectedly numeric
            if not isinstance(to_speak, str):
                to_speak = str(to_speak)

            # Cache key for audio bytes
            audio_key = f"{art['url']}_{st.session_state.language}_audio"

            # Button to trigger generation/playing
            # Use a unique key so Streamlit knows which article button was clicked
            play_btn_key = f"play_{i}_{hash(audio_key) & 0xffffffff}"

            # When user clicks the button, we generate audio (if not cached) and then show st.audio for playback
            if st.button(f"🔊 {translate_ui('Listen', st.session_state.language)}", key=play_btn_key):
                # Generate or fetch from cache
                audio_bytes = None
                if audio_key in st.session_state.audio_cache:
                    audio_bytes = st.session_state.audio_cache[audio_key]
                else:
                    # Generate via gTTS if available
                    if gTTS is not None:
                        try:
                            # chunk the text if very large to avoid service issues
                            # gTTS handles long strings but we be cautious: split into ~4000-character chunks
                            max_chunk = 4000
                            chunks = []
                            if len(to_speak) <= max_chunk:
                                chunks = [to_speak]
                            else:
                                sents = sent_tokenize(to_speak)
                                cur = ""
                                for s in sents:
                                    if len(cur) + len(s) + 1 < max_chunk:
                                        cur += s + " "
                                    else:
                                        chunks.append(cur.strip())
                                        cur = s + " "
                                if cur.strip():
                                    chunks.append(cur.strip())
                            # combine bytes
                            combined = io.BytesIO()
                            for idx, chunk in enumerate(chunks):
                                t = gTTS(text=chunk, lang=gtts_lang, slow=False)
                                temp_buf = io.BytesIO()
                                t.write_to_fp(temp_buf)
                                temp_buf.seek(0)
                                combined.write(temp_buf.read())
                            audio_bytes = combined.getvalue()
                            # cache in session
                            st.session_state.audio_cache[audio_key] = audio_bytes
                        except Exception as e:
                            try:
                                st.warning(f"Audio generation failed: {e}")
                            except:
                                pass
                            audio_bytes = None
                    else:
                        try:
                            st.warning("gTTS not available. Install gTTS package to enable audio playback.")
                        except:
                            pass
                        audio_bytes = None

                # If we have audio bytes, play them
                if audio_bytes:
                    # play using st.audio (Streamlit will serve bytes without saving file)
                    st.audio(audio_bytes, format='audio/mp3')
                else:
                    st.info("Audio not available for this article.")

            # Show read full link
            st.markdown(f"[{translate_ui('Read Full', st.session_state.language)} →]({art['url']})")
            st.markdown("---")

    else:
        st.warning(f"⚠️ No articles found. Try refreshing or selecting a different source.")
