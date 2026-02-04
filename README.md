# 🎬 Movie Recommendation System (Content-Based)

A Machine Learning based **Movie Recommendation System** built using **Python, Scikit-Learn, and Streamlit**.  
This system recommends similar movies based on metadata like cast, director, genre, and description using **TF-IDF vectorization** and **Cosine Similarity**.

It provides an interactive web interface where users can type a movie name and get top recommended movies instantly.

---

## 🚀 Features

- 🎥 Content-based recommendation
- 🧠 Uses TF-IDF vectorization
- 📐 Cosine similarity scoring
- ⚡ Fast similarity lookup
- 🖥 Interactive Streamlit web app
- 🔍 Partial title matching supported
- 📊 Uses real Netflix dataset

---

## 🧠 How Recommendation Works

The system uses **content-based filtering**:

1. Combine movie metadata:
   - Title
   - Director
   - Cast
   - Genre
   - Description

2. Convert text → numerical vectors using:

3. Compute similarity between all movies using:

4. When user enters a movie:
- Find matching movie
- Compare similarity scores
- Return Top N similar movies

---

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF Vectorizer
- Cosine Similarity

---

## 📂 Project Structure

Movie_Recommendation_System/
│
├── app.py
├── netflix_titles.csv
├── README.md


---

## 💻 Full Code (app.py)

```python
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("netflix_titles.csv")
df = df[['title', 'director', 'cast', 'listed_in', 'description']]
df.fillna('', inplace=True)

# Create metadata
df['metadata'] = (
    df['title'] + ' ' +
    df['director'] + ' ' +
    df['cast'] + ' ' +
    df['listed_in'] + ' ' +
    df['description']
).str.lower()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['metadata'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommandation(title, n=5):
    matches = df[df['title'].str.lower().str.contains(title.lower())]
    if matches.empty:
        return []
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.loc[movie_indices, 'title']

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Type a movie name and get similar recommendations")

movie = st.text_input("Enter movie name")

if st.button("Recommend"):
    if movie:
        results = recommandation(movie)
        if len(results) > 0:
            st.success("Recommended Movies:")
            for r in results:
                st.write("👉", r)
        else:
            st.error("Movie not found")
    else:
        st.warning("Please enter a movie name")

⚙️ Installation

git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

▶️ Run the App
streamlit run app.py
🧪 Example Usage
Input:
Inception

output:
👉 Interstellar
👉 Shutter Island
👉 The Prestige
👉 Tenet
👉 Memento
/screenshots/app_home.png
/screenshots/result.png
