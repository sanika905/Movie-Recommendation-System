
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
df = pd.read_csv("balanced_movie_dataset.csv")

# Data cleaning
df.drop_duplicates(subset='title', inplace=True)
df.dropna(subset=['overview', 'genre'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Combine overview and genre
df['content'] = df['overview'] + ' ' + df['genre']

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommend function with full info

def recommend(title, top_n=5):
    if title not in df['title'].values:
        return pd.DataFrame()

    idx = df[df['title'] == title].index[0]
    selected_language = df.loc[idx, 'language']

    sim_scores = list(enumerate(cosine_sim[idx]))

    # Filter: Only same-language movies
    sim_scores = [(i, score * df.iloc[i]['rating']) 
                  for i, score in sim_scores 
                  if i != idx and df.iloc[i]['language'] == selected_language]

    # Sort by similarity * rating
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    movie_indices = [i[0] for i in sim_scores]

    result_df = df.iloc[movie_indices][['title', 'genre', 'year', 'rating', 'language','time','industry']]
    result_df = result_df.sort_values(by='rating', ascending=False).reset_index(drop=True)
    return result_df


# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_list = df['title'].sort_values().unique()
selected_movie = st.selectbox("Select a movie you like:", movie_list)

if st.button("Recommend"):
    results = recommend(selected_movie)
    if not results.empty:
        st.subheader("Top Recommended Movies:")
        for idx, row in results.iterrows():
            st.markdown(f"**🎞️ {row['title']}**")
            st.markdown(f"• Genre: {row['genre']}")
            st.markdown(f"• Language: {row['language']}")   
            st.markdown(f"• Year: {row['year']}")  # make sure this matches your CSV
            st.markdown(f"• Rating: ⭐ {row['rating']}/10")
            st.markdown(f"• Time: 🕒 {row['time']}")
            st.markdown(f"• Industry: {row['industry']}") 
            
    else:
        st.warning("No recommendations found. Try another movie.")
