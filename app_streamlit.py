import streamlit as st
import pickle
import pandas as pd
import requests

# Load Data
movies_dict = pickle.load(open('movies_dic.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))

# TMDB Poster Fetch
API_KEY = "622a339467f3170bd68278872065e0e2"

@st.cache_data
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()

        poster_path = data.get('poster_path', None)

        if poster_path is None:
            return "https://via.placeholder.com/300x450?text=No+Poster"

        return "https://image.tmdb.org/t/p/w500/" + poster_path

    except:
        return "https://via.placeholder.com/300x450?text=Error"


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_names = []
    recommended_posters = []

    for i in movie_list:
        row = movies.iloc[i[0]]
        movie_id = row['movie_id'] if 'movie_id' in movies.columns else row['id']

        recommended_names.append(row['title'])
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_names, recommended_posters
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])
