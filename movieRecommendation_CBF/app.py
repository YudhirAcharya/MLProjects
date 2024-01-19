import streamlit as st
import pickle
movies = pickle.load(open("movies_list.pkl", 'rb'))
st.header("Movie Recommender Application")

