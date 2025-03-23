import streamlit as st
import sqlite3
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to clean text
def clean_text(text):
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = word_tokenize(clean_text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(clean_tokens).strip()

# Function to fetch documents from SQLite
def fetch_document_data():
    conn = sqlite3.connect("chroma.sqlite3")  # Update with your DB path
    c = conn.cursor()
    c.execute("SELECT * FROM embedding_fulltext_search_content")
    document_data = c.fetchall()
    conn.close()
    return document_data

# Streamlit UI
st.title("Semantic Search Engine")

# User input
search_query = st.text_input("Enter your search query:")

if st.button("Search"):
    if search_query:
        cleaned_query = clean_text(search_query)
        query_embedding = model.encode([cleaned_query])

        # Fetch documents from SQLite
        document_data = fetch_document_data()
        
        similarities = []
        for doc_data in document_data:
            document_text = doc_data[1]  # Assuming content is in the second column
            document_embedding = model.encode([document_text])
            similarity = cosine_similarity(query_embedding, document_embedding)[0][0]
            similarities.append((document_text, similarity))
        
        # Sort results by similarity score
        sorted_documents = sorted(similarities, key=lambda x: x[1], reverse=True)

        # Display top results
        st.subheader("Top 10 Related Subtitles:")
        for i, (content, _) in enumerate(sorted_documents[:10], 1):
            st.write(f"**{i}.** {content}")

    else:
        st.warning("Please enter a search query.")
