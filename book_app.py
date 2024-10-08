# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:57:22 2024

@author: admin
"""
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load the data
data = pd.read_csv('final_ratings.csv')

# Create a pivot table for book ratings
book_pivot = data.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)

# Convert the pivot table to a sparse matrix
book_sparse = csr_matrix(book_pivot)

# Create and fit the Nearest Neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(book_sparse)

# Save the model to a pickle file
with open('nearest_neighbors_model.pkl', 'wb') as file:
    pickle.dump(model, file)
    
    
    
import streamlit as st
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load the model from the pickle file
with open('nearest_neighbors_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the data
data = pd.read_csv('final_ratings.csv')

# Create a pivot table for book ratings
book_pivot = data.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)

# Convert the pivot table to a sparse matrix
book_sparse = csr_matrix(book_pivot)

# Prepare the DataFrame for book titles
book_df = data[['Book-Title']].drop_duplicates()

# Function to get recommendations
def get_recommendations(book_title):
    # Get the index of the book
    book_index = book_df[book_df['Book-Title'] == book_title].index[0]
    
    # Find the nearest neighbors
    distances, indices = model.kneighbors(book_sparse[book_index], n_neighbors=6)  # 6 to include the book itself
    
    # Return book titles, excluding the selected book
    return book_df.iloc[indices[0][1:]]['Book-Title'].values

# Streamlit app layout
st.title("Book Recommendation System")

# User input for book title
book_title = st.selectbox("Select a book title:", book_df['Book-Title'].values)

# Recommend books when button is clicked
if st.button("Recommend"):
    recommendations = get_recommendations(book_title)
    st.write("### Recommended Books:")
    for rec in recommendations:
        st.write(rec)
