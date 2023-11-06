import pickle
import numpy as np
import streamlit as st
import requests
from sklearn.neighbors import NearestNeighbors
st.set_page_config(layout="wide")
def recommend(movie):
    l=[]
    index1=movies[movies['Name of movie'] == movie].index[0]
    k = np.array(embeddings1[index1]).tolist()
    l.append(k)
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(embeddings)
    neighbors = nn.kneighbors(l, return_distance=False)[0]
    ki1= movies['Name of movie'].iloc[neighbors].tolist()
    return ki1

st.header('Movie Recommender System')
movies = pickle.load(open('Name of movie.pkl','rb'))
embeddings= pickle.load(open('embeddings.pkl','rb'))
embeddings1= pickle.load(open('embeddings1.pkl','rb'))
Link= pickle.load(open('Link.pkl','rb'))
movie_list = movies['Name of movie'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

def img(value):
    lis1=[]
    lis=Link[Link['Name of movie']==value]['Link']
    lis1=lis.tolist()
    if len(lis1)>1:
        return lis1[0]
    else:
        return lis1


if st.button('Show Recommendation'):
    ki = recommend(selected_movie)
    col1, col2, col3, col4, col5,col6, col7, col8, col9, col10 = st.beta_columns(10)
    with col1:
        st.image(img(ki[0]), width=120, use_column_width=120, clamp=False, channels="RGB", output_format="auto")
        st.write(f"**{ki[0]}**")
    with col2:

        st.image(img(ki[1]), width=120, use_column_width=120, clamp=False, channels="RGB", output_format="auto")
        st.write(f"**{ki[1]}**")
    with col3:
        st.image(img(ki[2]), width=120, use_column_width=120, clamp=False, channels="RGB", output_format="auto")
        st.write(f"**{ki[2]}**")

    with col4:
        st.image(img(ki[3]), width=120, use_column_width=120, clamp=False, channels="RGB", output_format="auto")
        st.write(f"**{ki[3]}**")

    with col5:
        st.image(img(ki[4]), width=120, use_column_width=120, clamp=False, channels="RGB", output_format="auto")
        st.write(f"**{ki[4]}**")
    with col6:
        st.image(img(ki[5]), width=120, use_column_width=120, clamp=False, channels="RGB", output_format="auto")
        st.write(f"**{ki[5]}**")
    with col7:

        st.image(img(ki[6]), width=120, use_column_width=120, clamp=False, channels="RGB", output_format="auto")
        st.write(f"**{ki[6]}**")
    with col8:
        st.image(img(ki[7]), width=120, use_column_width=120, clamp=False, channels="RGB", output_format="auto")
        st.write(f"**{ki[7]}**")

    with col9:
        st.image(img(ki[8]), width=120, use_column_width=100, clamp=False, channels="RGB", output_format="auto")
        st.write(f"**{ki[8]}**")

    with col10:
        st.image(img(ki[9]), width=120, use_column_width=120, clamp=False, channels="RGB", output_format="auto")
        st.write(f"**{ki[9]}**")






