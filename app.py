import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

@st.cache_resource
def load():
    movies = pd.read_csv('data/movies.dat', sep='::', engine='python',
                         names=['movie_id', 'title', 'genres'], encoding='latin-1')
    ratings = pd.read_csv('data/ratings.dat', sep='::', engine='python',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
    ids = ratings['movie_id'].astype('category')
    item_ids = ids.cat.categories
    counts = ratings.groupby('movie_id').size()
    m = np.load('results/svd_model.npz')
    V = m['V']
    movies = movies.set_index('movie_id').loc[item_ids]
    movies['n_ratings'] = counts.loc[item_ids].values
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return movies.reset_index(), Vn

movies, Vn = load()

st.title("🎬 Movie Recommender")
st.caption("Matrix factorization (SVD trained from scratch with NumPy) on MovieLens 1M")

title = st.selectbox("Pick a movie you like:", sorted(movies['title']))
n = st.slider("How many recommendations?", 5, 15, 10)

if st.button("Recommend"):
    idx = movies.index[movies['title'] == title][0]
    sim = Vn @ Vn[idx]
    sim[movies['n_ratings'].values < 50] = -1
    top = np.argsort(-sim)
    top = [i for i in top if i != idx][:n]
    st.subheader("You might also like:")
    for i in top:
        st.write(f"**{movies.loc[i, 'title']}**  \n*{movies.loc[i, 'genres']}*")