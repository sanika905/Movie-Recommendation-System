import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
df = pd.read_csv("balanced_movie_dataset_with_posters.csv")

# Data Cleaning
df.drop_duplicates(subset='title', inplace=True)
df.dropna(subset=['overview', 'genre'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Combine overview + genre + language + industry for content
df['content'] = df['overview'] + ' ' + df['genre'] + ' ' + df['language'] + ' ' + df['industry']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Similarity Matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation Function
def recommend(title, top_n=5):
    if title not in df['title'].values:
        return pd.DataFrame()
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'genre', 'language', 'industry', 'year', 'rating', 'time', 'poster_url']]

# Streamlit UI
st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

movie_list = df['title'].sort_values().unique()
selected_movie = st.selectbox("🎞️ Select a movie you like:", movie_list)

if st.button("🔍 Recommend"):
    results = recommend(selected_movie)
    if not results.empty:
        st.subheader("🎯 Top Recommended Movies:")
        for idx, row in results.iterrows():
            st.markdown(f"### 🎥 {row['title']}")

            # Poster handling: if poster_url is invalid, show placeholder
            poster = str(row['poster_url'])
            if not poster.startswith("http"):
                poster = "https://via.placeholder.com/200x300.png?text=No+Image"
            st.image(poster, width=200)

            st.markdown(f"• **Genre:** {row['genre']}")
            st.markdown(f"• **Language:** {row['language']}")
            st.markdown(f"• **Industry:** {row['industry']}")
            st.markdown(f"• **Year:** {row['year']}")
            st.markdown(f"• **Rating:** ⭐ {row['rating']}/10")
            st.markdown(f"• **Time:** 🕒 {row['time']}")
            st.markdown("---")
    else:
        st.warning("No recommendations found. Try another movie.")
