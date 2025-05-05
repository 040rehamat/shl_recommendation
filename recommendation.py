# recommendation.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load catalog
df = pd.read_csv('shl_catalog.csv')

# Combine fields into one text column
df['combined'] = df['role'] + " " + df['skills'] + " " + df['industry']

# Load model (pretrained sentence embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for catalog items
catalog_embeddings = model.encode(df['combined'].tolist())

def get_recommendations(user_input, top_k=5):
    # Embed user query
    query_embedding = model.encode([user_input])

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, catalog_embeddings)[0]

    # Get top_k indices
    top_indices = similarities.argsort()[::-1][:top_k]

    # Return top recommendations
    recommendations = df.iloc[top_indices][['assessment_name', 'role', 'skills', 'industry']]
    recommendations['similarity'] = similarities[top_indices].round(3)
    return recommendations.to_dict(orient='records')
