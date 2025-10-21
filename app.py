# app.py  — full Streamlit app with gTTS voice playback on click (female/newscaster style)
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
import base64
import io
import importlib.util
import subprocess
import sys

# Try auto-install packages if missing (deep_translator, gtts)
def try_install(package_name, import_name=None):
    if importlib.util.find_spec(import_name or package_name) is None:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        except Exception as e:
            try:
                st.warning(f"Auto-install failed for {package_name}: {e}")
            except:
                pass

try_install("deep-translator", "deep_translator")
try_install("gTTS", "gtts")

# Now import translator and gTTS if available
try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

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
        nltk.data.find(f'corpora/{c}' if c=='stopwords' else f'tokenizers/{c}')
    except:
        nltk.download(c, quiet=True)

# Languages
LANGUAGES = {
    "🇬🇧 English": ("en", "en-US"),
    "🇮🇳 हिंदी": ("hi", "hi-IN"),
    "🇮🇳 मराठी": ("mr", "mr-IN"),
    "🇮🇳 ಕನ್ನಡ": ("kn", "kn-IN")
}

# Translation dictionaries
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

@st.cache_data(ttl=3600)
def translate_text(text, target_lang):
    """Translate text using deep-translator (cached)"""
    if GoogleTranslator is None:
        return text
    if not text or target_lang == "en":
        return text
    try:
        max_length = 4500
        if len(text) <= max_length:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
            return translated
        else:
            sentences = sent_tokenize(text)
            translated_sentences = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_length:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        trans = GoogleTranslator(source='auto', target=target_lang).translate(current_chunk)
                        translated_sentences.append(trans)
                    current_chunk = sentence + " "
            if current_chunk:
                trans = GoogleTranslator(source='auto', target=target_lang).translate(current_chunk)
                translated_sentences.append(trans)
            return " ".join(translated_sentences)
    except Exception as e:
        st.warning(f"Translation error: {e}")
        return text

def translate_ui(text, lang_code):
    """Translate UI elements"""
    if lang_code == "en":
        return text
    return TRANSLATIONS.get(lang_code, {}).get(text, text)

# Categories (same as your original)
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

def clean_title(title, content):
    """Clean noisy news site titles but don't overwrite real titles"""
    t = (title or "").strip()
    for s in ['BBC', 'Reuters', 'CNN', 'Hindu', 'NDTV', 'ESPN', 'Variety', 'TechCrunch', 'Bloomberg', 'Guardian']:
        t = t.replace(f' - {s}', '').replace(f'| {s}', '').replace(f': {s}', '')
    # remove duplicates like 'NewsNews' and filler words
    t = t.replace('NewsNews', '').replace('Latest', '').replace('Breaking', '')
    t = ' '.join(t.split())
    if not t or len(t) < 10:
        sents = sent_tokenize(content or "")
        if sents:
            t = sents[0].split(',')[0] if ',' in sents[0] else sents[0]
    if len(t) > 120:
        t = t.split(' - ')[0] if ' - ' in t else ' '.join(t.split()[:18]) + '...'
    if t.isupper():
        t = t.title()
    return t.strip(' .-:') or "Article"

def get_keywords(text, n=5):
    try:
        t = (text or "").lower()
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

def fetch_article(url):
    h = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        r = requests.get(url, headers=h, timeout=12)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'html.parser')
        for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
            t.decompose()
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else None
        text = ""
        arts = soup.find_all(['article', 'div', 'main'],
                             class_=lambda x: x and any(k in str(x).lower() for k in ['article', 'story', 'content', 'post', 'body']))
        if arts:
            for a in arts[:3]:
                ps = a.find_all('p')
                for p in ps:
                    pt = p.get_text(strip=True)
                    if len(pt) > 30:
                        text += pt + ' '
                if len(text) > 300:
                    break
        if len(text) < 200:
            ps = soup.find_all('p')
            for p in ps:
                pt = p.get_text(strip=True)
                if len(pt) > 30:
                    text += pt + ' '
                if len(text) > 500:
                    break
        text = ' '.join(text.split())
        # If we couldn't find a title, try to craft from content (fallback)
        if (not title or title.strip() == '') and text:
            title = clean_title("", text)
        if text and len(text) > 150:
            return {'title': title or "Article", 'url': url, 'content': text, 'time': datetime.now()}
    except Exception:
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
                # ensure title is cleaned but keep fetched title intact
                fetched += 1
                articles.append(a)
    except Exception:
        pass
    return articles

def summarize(text, n=6):
    """Enhanced summarization with better sentence selection"""
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

# Audio generation using gTTS -> returns base64 mp3 string
def generate_tts_base64(text, lang_code):
    """
    Generate mp3 bytes using gTTS and return base64 string.
    lang_code: short code like 'en','hi','mr','kn' — will fallback gracefully.
    """
    if gTTS is None:
        return None
    try:
        # gTTS may not support all language codes exactly; use mapping/fallbacks if needed
        gtts_lang = lang_code
        # For Kannada, gTTS uses 'kn' sometimes unsupported — fallback to English
        if lang_code == 'kn':
            # gTTS may not have Kannada voice on some installs; fallback to 'en' for reliability
            try:
                # try creating with kn; if fails, will go to except
                buf_test = io.BytesIO()
                gTTS(" ", lang=lang_code).write_to_fp(buf_test)
            except Exception:
                gtts_lang = 'en'
        # Create a BytesIO buffer and write mp3 into it
        buf = io.BytesIO()
        # Use a pacing technique: ensure punctuation and short breaks for anchor-style reading
        # The caller should add punctuation appropriately.
        tts = gTTS(text.strip(), lang=gtts_lang, slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        return b64
    except Exception as e:
        # if audio generation fails, return None and warn
        try:
            st.warning(f"Audio generation failed: {e}")
        except:
            pass
        return None

# Streamlit App
st.set_page_config(layout="wide", page_title="Insight Ink Pro", page_icon="🚀", initial_sidebar_state="expanded")
add_pwa_support()

# CSS (kept as your original)
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
    if new_lang != st.session_state.prev_language:
        st.session_state.language = new_lang
        st.session_state.prev_language = new_lang
        st.session_state.translated_articles = {}
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
                st.rerun()

else:
    if st.session_state.articles:
        st.success(f"✅ {len(st.session_state.articles)} {translate_ui('articles loaded', st.session_state.language)}")
        for i, art in enumerate(st.session_state.articles):
            # Use fetched title (do not override) but clean it for fallback if necessary
            fetched_title = art.get('title', '') or clean_title('', art.get('content', ''))
            # Summarize content only
            summary_en = summarize(art.get('content', ''), 6)
            # Translate title and summary if needed
            lang_code = st.session_state.language
            title_trans = fetched_title
            summary_trans = summary_en
            cache_key_title = f"{art['url']}_{lang_code}_title"
            cache_key_summary = f"{art['url']}_{lang_code}_summary"
            if lang_code != "en":
                # translate title
                if cache_key_title not in st.session_state.translated_articles:
                    with st.spinner(f"🌐 Translating title {i+1}..."):
                        t_title = translate_text(fetched_title, lang_code)
                        st.session_state.translated_articles[cache_key_title] = {'title': t_title}
                title_trans = st.session_state.translated_articles[cache_key_title]['title']
                # translate summary
                if cache_key_summary not in st.session_state.translated_articles:
                    with st.spinner(f"🌐 Translating article {i+1} summary..."):
                        t_sum = translate_text(summary_en, lang_code)
                        st.session_state.translated_articles[cache_key_summary] = {'summary': t_sum}
                summary_trans = st.session_state.translated_articles[cache_key_summary]['summary']
            # Keywords
            keywords = get_keywords(art.get('content', ''), 5)
            # Display
            col1, col2 = st.columns([5, 1])
            with col1:
                # show original fetched title but show translated title below in smaller text (if language != en)
                st.markdown(f"### 📌 {fetched_title}")
                if lang_code != "en" and title_trans and title_trans != fetched_title:
                    st.markdown(f"**Translated:** {title_trans}")
            with col2:
                if i < 5:
                    st.markdown('<span class="new-badge">NEW</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="time-badge">{time_ago(art["time"])}</span>', unsafe_allow_html=True)
            if keywords:
                kw_html = ''.join([f'<span class="keyword-tag">{k}</span>' for k in keywords])
                st.markdown(kw_html, unsafe_allow_html=True)
            st.markdown(f"**✨ {translate_ui('Quick Summary', st.session_state.language)}:**")
            # Highlighted summary
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
                    border-left: 4px solid #6366f1;
                    padding: 12px;
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
            # Prepare anchor-style text for TTS: Title + short pause + summary
            # Add punctuation / pauses to sound like an anchor
            # E.g. "Title. — Now, the summary in concise form."
            combined_for_tts = f"{title_trans}. — {summary_trans}"
            # Generate audio base64 once (cached by session_state ideally)
            audio_cache_key = f"{art['url']}_{lang_code}_audio_b64"
            audio_b64 = st.session_state.translated_articles.get(audio_cache_key, {}).get('b64') if audio_cache_key in st.session_state.translated_articles else None
            if audio_b64 is None:
                audio_b64 = generate_tts_base64(combined_for_tts, lang_code)
                # store in translations cache dict under audio cache key
                if audio_b64:
                    st.session_state.translated_articles[audio_cache_key] = {'b64': audio_b64}
            # Render Listen button + hidden audio player that only plays on click
            btn_id = f"playbtn_{i}"
            audio_id = f"audio_{i}"
            if audio_b64:
                # embed a tiny JS widget: hidden audio element and a visible button. On click toggle play/pause.
                audio_html = f"""
                <div style="margin-top:8px;">
                  <button id="{btn_id}" class="voice-btn">🎙️ {translate_ui('Listen', st.session_state.language)}</button>
                </div>
                <audio id="{audio_id}" preload="auto" style="display:none;">
                  <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                </audio>
                <script>
                (function() {{
                  const btn = document.getElementById('{btn_id}');
                  const audio = document.getElementById('{audio_id}');
                  let isPlaying = false;
                  // toggle behavior but ensure user click triggers actual playback
                  btn.addEventListener('click', function() {{
                    if (!audio) {{
                      alert('Audio not available');
                      return;
                    }}
                    if (audio.paused) {{
                      audio.currentTime = 0;
                      audio.play().catch(e => {{
                        console.warn('play failed', e);
                        alert('Playback failed: user gesture required or browser blocked it.');
                      }});
                      btn.innerText = '⏸️ Stop';
                      btn.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
                    }} else {{
                      audio.pause();
                      audio.currentTime = 0;
                      btn.innerText = '🎙️ {translate_ui('Listen', st.session_state.language)}';
                      btn.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
                    }}
                  }});
                  audio.onended = function() {{
                    btn.innerText = '🎙️ {translate_ui('Listen', st.session_state.language)}';
                    btn.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
                  }};
                }})();
                </script>
                """
                components.html(audio_html, height=100)
            else:
                # audio generation failed fallback: provide a JS TTS using speechSynthesis as backup
                # (speechSynthesis can be lower quality and may not have female voice for given lang)
                js_text = combined_for_tts.replace("`", "\\`").replace("\n", " ")
                fallback_html = f"""
                <div style="margin-top:8px;">
                  <button id="{btn_id}" class="voice-btn">🎙️ {translate_ui('Listen', st.session_state.language)}</button>
                </div>
                <script>
                (function() {{
                  const btn = document.getElementById('{btn_id}');
                  let utter = null;
                  btn.addEventListener('click', function() {{
                    if (!('speechSynthesis' in window)) {{
                      alert('Speech synthesis not supported in your browser.');
                      return;
                    }}
                    if (utter && speechSynthesis.speaking) {{
                      speechSynthesis.cancel();
                      btn.innerText = '🎙️ {translate_ui('Listen', st.session_state.language)}';
                      btn.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
                      return;
                    }}
                    const text = `{js_text}`;
                    utter = new SpeechSynthesisUtterance(text);
                    utter.lang = '{lang_code}';
                    utter.rate = 0.95;
                    utter.pitch = 1.0;
                    const voices = speechSynthesis.getVoices();
                    // attempt to pick a female voice name
                    const female = voices.find(v => /female|zira|samantha|victoria|alloy|google|woman/i.test(v.name) && v.lang && v.lang.startsWith('{lang_code}'.split('-')[0]));
                    if (female) utter.voice = female;
                    speechSynthesis.cancel();
                    speechSynthesis.speak(utter);
                    btn.innerText = '⏸️ Stop';
                    btn.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
                    utter.onend = function() {{
                      btn.innerText = '🎙️ {translate_ui('Listen', st.session_state.language)}';
                      btn.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
                    }};
                    utter.onerror = function() {{
                      btn.innerText = '🎙️ {translate_ui('Listen', st.session_state.language)}';
                      btn.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
                    }};
                  }});
                }})();
                </script>
                """
                components.html(fallback_html, height=100)
            st.markdown(f"[{translate_ui('Read Full', st.session_state.language)} →]({art['url']})")
            st.markdown("---")
    else:
        st.warning(f"⚠️ No articles found. Try refreshing or selecting a different source.")
