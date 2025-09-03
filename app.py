import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# --- NLTK Data Download (Run this once if you haven't) ---
# Using a more general Exception catch for robustness across NLTK versions
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')
# --- Download punkt_tab if not found ---
try:
    nltk.data.find('tokenizers/punkt_tab') # Punkt tokenizer tables
except Exception:
    nltk.download('punkt_tab')

# --- Configuration ---
NEWS_SOURCES = {
    "BBC News (Technology)": "https://www.bbc.com/news/technology",
    "Reuters (World)": "https://www.reuters.com/world/",
    "The Hindu (India)": "https://www.thehindu.com/latest-news/",
    "Yahoo Finance (News)": "https://finance.yahoo.com/news/" # Added a financial news source
}

NUM_TOPICS = 5  # Number of topics to extract
NUM_TOP_WORDS = 5 # Number of top words to show for each topic
SUMMARY_SENTENCES = 3 # Number of sentences for extractive summary
MAX_ARTICLES_TO_PROCESS = 5 # Limit the number of individual articles to process for speed from predefined sources

# --- Helper Functions ---

def fetch_single_article_content(url):
    """Fetches the main text content and title from a single given URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title (common tags: h1, title)
        title_tag = soup.find('h1') or soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else "No Title Found"

        # Attempt to find common article content containers
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

        if not article_text:
            # Fallback for some sites, try to get text from body
            article_text = soup.body.get_text(separator=' ', strip=True)
            # Clean up excessive whitespace
            article_text = re.sub(r'\s+', ' ', article_text).strip()

        if article_text and len(article_text) > 100: # Ensure some minimum content
            return {'title': title, 'url': url, 'content': article_text}
        else:
            st.warning(f"Could not extract sufficient content from: {url}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching {url}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while parsing {url}: {e}")
        return None

def fetch_article_links_and_content(main_url):
    """
    Fetches the main page, extracts individual article links,
    and then fetches content for each linked article.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    articles_data = []

    try:
        st.info(f"Visiting main page: {main_url}")
        main_response = requests.get(main_url, headers=headers, timeout=10)
        main_response.raise_for_status()
        main_soup = BeautifulSoup(main_response.text, 'html.parser')

        # Find potential article links on the main page
        links = set() # Use a set to avoid duplicate URLs
        
        for tag in main_soup.find_all(['h2', 'h3', 'a'], class_=lambda x: x and ('headline' in x.lower() or 'article' in x.lower() or 'story' in x.lower() or 'news' in x.lower())):
            if tag.name == 'a' and tag.get('href'):
                link = tag['href']
            elif tag.find('a', href=True):
                link = tag.find('a')['href']
            else:
                continue

            if link.startswith('http'):
                links.add(link)
            elif link.startswith('/'):
                base_url_parts = main_url.split('/')
                if len(base_url_parts) > 2:
                    base_url = '/'.join(base_url_parts[:3])
                    links.add(base_url + link)
            
            if len(links) >= MAX_ARTICLES_TO_PROCESS * 2: # Scrape more links than we need, then filter
                break

        st.info(f"Found {len(links)} potential article links on the main page. Fetching content for up to {MAX_ARTICLES_TO_PROCESS}...")
        
        articles_fetched_count = 0
        for link in list(links):
            if articles_fetched_count >= MAX_ARTICLES_TO_PROCESS:
                break
            
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Fetching: {link[:70]}...")
            article_data = fetch_single_article_content(link) # Reuse fetch_single_article_content
            if article_data:
                articles_data.append(article_data)
                articles_fetched_count += 1

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching main page or articles from {main_url}: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while parsing {main_url}: {e}")
    
    return articles_data

def preprocess_text(text):
    """Cleans and tokenizes text, removing stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def train_lda_model(documents, num_topics, num_top_words):
    """Trains an LDA model and returns topics."""
    if not documents:
        return None, None, []

    # Use CountVectorizer for LDA
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english') # Changed to TfidfVectorizer for consistency
    dtm = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        learning_method='batch',
        max_iter=20
    )
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    
    return lda, dtm, topics

def summarize_article(text, num_sentences=3):
    """Generates an extractive summary of an article using TF-IDF and cosine similarity."""
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        # Fit on sentences and transform them
        sentence_vectors = vectorizer.fit_transform(sentences)
        
        # Transform the entire document (concatenated sentences) into a single vector
        # Ensure consistent tokenization/preprocessing for document vector
        document_vector = vectorizer.transform([' '.join(sentences)])

        # Calculate cosine similarity between each sentence and the entire document
        # Reshape document_vector if it's 1D to (1, -1) for cosine_similarity
        if document_vector.ndim == 1:
            document_vector = document_vector.reshape(1, -1)
            
        sentence_scores = cosine_similarity(sentence_vectors, document_vector).flatten()

        # Get indices of top sentences based on score
        top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        
        # Sort these indices to maintain original sentence order
        sorted_indices = sorted(top_sentence_indices.tolist())
        
        summary = ' '.join([sentences[i] for i in sorted_indices])
        return summary
    except ValueError:
        st.warning("Could not create TF-IDF matrix for summarization. Returning first sentences.")
        return ' '.join(sentences[:num_sentences])
    except Exception as e:
        st.error(f"Error during summarization: {e}. Returning first sentences.")
        return ' '.join(sentences[:num_sentences])

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Personalized News Summarizer")

st.title("📰 Personalized News Summarizer with Topic Modeling")
st.markdown("---")

st.sidebar.header("Settings")

# Option to use custom URL
use_custom_url = st.sidebar.checkbox("Summarize a specific article URL?", False)
custom_url = ""
if use_custom_url:
    custom_url = st.sidebar.text_input("Enter Article URL:", "")
    selected_source = None # Clear predefined source selection
else:
    selected_source = st.sidebar.selectbox("Select News Source:", list(NEWS_SOURCES.keys()))
    st.sidebar.info("Note: Web scraping can be unreliable if website structures change.")

# User input for preferred topics
st.sidebar.subheader("Your Preferred Topics (Keywords)")
preferred_topics_input = st.sidebar.text_input(
    "Enter keywords separated by commas (e.g., 'tech, AI, science, stock')",
    value="tech, AI, stock"
)
preferred_keywords = [kw.strip().lower() for kw in preferred_topics_input.split(',') if kw.strip()]

if st.sidebar.button("Fetch & Analyze News"):
    if use_custom_url and not custom_url:
        st.warning("Please enter a custom article URL or uncheck the option.")
    elif not use_custom_url and not selected_source:
        st.warning("Please select a news source.")
    else:
        articles_to_process = []
        if use_custom_url:
            st.info(f"Fetching content from custom URL: {custom_url}...")
            with st.spinner("Fetching and processing single article..."):
                article_data = fetch_single_article_content(custom_url)
                if article_data:
                    articles_to_process.append(article_data)
        else:
            st.info(f"Fetching news from {selected_source}...")
            url_to_scrape = NEWS_SOURCES[selected_source]
            with st.spinner("Fetching and processing articles..."):
                articles_to_process = fetch_article_links_and_content(url_to_scrape)

        if articles_to_process:
            if use_custom_url: # Display for a single custom URL
                article = articles_to_process[0]
                st.success(f"Successfully fetched and processed article: {article['title']}")
                st.subheader(f"Summary and Topics for: {article['title']}")
                
                # Preprocess for topic modeling and summarization
                # For LDA, we usually want a collection of documents (or sentences treated as documents)
                # For summarization, we work on the original sentences
                
                summary = summarize_article(article['content'], SUMMARY_SENTENCES)
                
                # For LDA, use sentences as individual documents
                sentences_for_lda = sent_tokenize(article['content'])
                # Preprocess each sentence for LDA
                processed_sentences_for_lda = [preprocess_text(s) for s in sentences_for_lda if preprocess_text(s)]

                lda_model, count_vectorizer_dtm, topics_list = train_lda_model(
                    processed_sentences_for_lda, NUM_TOPICS, NUM_TOP_WORDS
                )

                st.markdown("---")
                st.subheader("Summarized Content:")
                st.write(summary)
                st.markdown("---")

                if topics_list:
                    st.subheader("Identified Topics:")
                    for topic in topics_list:
                        st.write(f"- {topic}")
                    st.markdown("---")
                
                st.markdown(f"**Original Article Link:** [Read more]({article['url']})")

                # Check for keyword matches in summary and topics for the single article
                summary_words = word_tokenize(summary.lower())
                summary_matches = [kw for kw in preferred_keywords if kw in summary_words]

                topic_words_flat = []
                if topics_list:
                    for topic_str in topics_list:
                        match = re.search(r'Topic \d+: (.*)', topic_str)
                        if match:
                            topic_words_flat.extend([w.strip().lower() for w in match.group(1).split(',')])
                topic_matches = [kw for kw in preferred_keywords if kw in topic_words_flat]

                if summary_matches or topic_matches:
                    st.success(f"**MATCH!** This article is relevant to your preferred topics.")
                    if summary_matches:
                        st.write(f"Keywords found in summary: {', '.join(summary_matches)}")
                    if topic_matches:
                        st.write(f"Keywords found in identified topics: {', '.join(topic_matches)}")
                else:
                    st.info("No direct matches found for your preferred topics in this article.")

            else: # Display for multiple articles from predefined sources
                st.success(f"Successfully fetched and processed {len(articles_to_process)} articles!")
                st.subheader("Personalized News Feed:")
                matched_articles_count = 0

                for i, article in enumerate(articles_to_process):
                    st.markdown(f"---")
                    st.markdown(f"**Article {i+1}: {article['title']}**")
                    
                    summary = summarize_article(article['content'], SUMMARY_SENTENCES)
                    
                    sentences_for_lda = sent_tokenize(article['content'])
                    processed_sentences_for_lda = [preprocess_text(s) for s in sentences_for_lda if preprocess_text(s)]
                    
                    lda_model, count_vectorizer_dtm, topics_list = train_lda_model(
                        processed_sentences_for_lda, NUM_TOPICS, NUM_TOP_WORDS
                    )

                    summary_words = word_tokenize(summary.lower())
                    summary_matches = [kw for kw in preferred_keywords if kw in summary_words]

                    topic_words_flat = []
                    if topics_list:
                        for topic_str in topics_list:
                            match = re.search(r'Topic \d+: (.*)', topic_str)
                            if match:
                                topic_words_flat.extend([w.strip().lower() for w in match.group(1).split(',')])
                    topic_matches = [kw for kw in preferred_keywords if kw in topic_words_flat]

                    if summary_matches or topic_matches:
                        matched_articles_count += 1
                        st.success(f"**MATCH!** This article is relevant to your preferred topics.")
                        st.write(f"**Summary:** {summary}")
                        if topics_list:
                            st.write("**Identified Topics:**")
                            for topic in topics_list:
                                st.write(f"- {topic}")
                        if summary_matches:
                            st.write(f"Keywords found in summary: {', '.join(summary_matches)}")
                        if topic_matches:
                            st.write(f"Keywords found in identified topics: {', '.join(topic_matches)}")
                        st.markdown(f"**Full Article Link:** [Read more]({article['url']})")
                    else:
                        st.info(f"This article does not directly match your preferred topics. (Summary: {summary[:100]}...)")
                        st.markdown(f"**Full Article Link:** [Read more]({article['url']})")

                if matched_articles_count == 0:
                    st.warning("No articles found matching your preferred topics from the selected source's main page.")
                else:
                    st.success(f"Displayed {matched_articles_count} relevant articles.")

        else:
            st.error("Could not retrieve any articles. This might be due to web scraping limitations or no articles found.")

st.markdown("---")