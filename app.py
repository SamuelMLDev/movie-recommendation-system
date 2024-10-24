from flask import Flask, jsonify, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the functions and data from movie_recommender.py
from movie_recommender import get_recommendations, recommend_popular_movies

@app.route('/')
def home():
    return render_template('index.html')  # Serve the front-end HTML

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    num_recommendations = int(request.args.get('num_recommendations', 5))  # Default to 5 recommendations
    rating_threshold = float(request.args.get('rating_threshold', 3.5))  # Default rating threshold of 3.5
    similar_users_count = int(request.args.get('similar_users_count', 10))  # Default 10 similar users

    recommendations = get_recommendations(user_id=user_id, num_recommendations=num_recommendations, 
                                          rating_threshold=rating_threshold, similar_users_count=similar_users_count)
    return jsonify(recommendations.index.tolist())

@app.route('/popular', methods=['GET'])
def popular():
    popular_movies = recommend_popular_movies(num_recommendations=5)
    return jsonify(popular_movies.index.tolist())

if __name__ == '__main__':
    app.run(debug=True)
