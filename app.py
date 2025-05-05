from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the product catalog CSV
df = pd.read_csv('shl_catalog.csv')  # Ensure this CSV is in same folder

# Combine columns into a searchable text field
df['combined'] = df['role'] + ' ' + df['skills'] + ' ' + df['industry']

# Prepare TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_input = data.get('user_input', '')
    user_vec = vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()

    top_indices = cosine_sim.argsort()[-5:][::-1]  # Top 5
    recommendations = []
    for idx in top_indices:
        recommendations.append({
            'Assessment Name': df.iloc[idx]['assessment_name'],
            'Role': df.iloc[idx]['role'],
            'Skills': df.iloc[idx]['skills'],
            'Industry': df.iloc[idx]['industry'],
            'Similarity': round(float(cosine_sim[idx]), 3)
        })
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
