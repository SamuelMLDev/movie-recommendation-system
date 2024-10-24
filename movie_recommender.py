import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the ratings data
column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)

# Load the movie data
movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 
                 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=movie_columns, encoding='latin-1')

# Merge the datasets on the movie_id column
merged_data = pd.merge(ratings, movies[['movie_id', 'title']], on='movie_id')

# Create a user-movie matrix where rows are user_ids and columns are movie titles
user_movie_matrix = merged_data.pivot_table(index='user_id', columns='title', values='rating')

# Fill NaN values with 0, since a missing rating implies the user hasn't rated that movie
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix_filled)

# Convert the result into a DataFrame for easier manipulation
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# A simple function to get a user's favorite genres based on the movies they've rated highly
def get_favorite_genres(user_id, rating_threshold=4.0):
    # Get the movies the user has rated highly
    high_rated_movies = merged_data[(merged_data['user_id'] == user_id) & (merged_data['rating'] >= rating_threshold)]
    
    # Get the genres of these movies
    favorite_genres = movies[movies['title'].isin(high_rated_movies['title'])].iloc[:, 5:]  # genre columns start from index 5
    genre_counts = favorite_genres.sum().sort_values(ascending=False)
    
    return genre_counts.index[:3]  # Return top 3 favorite genres

# Function to get movie recommendations for a user
def get_recommendations(user_id, num_recommendations=5, rating_threshold=3.5, similar_users_count=10):
    # Get the most similar users to the target user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:similar_users_count+1]
    
    # Get the movie ratings of the most similar users
    similar_user_ratings = user_movie_matrix.loc[similar_users]
    
    # Calculate the average ratings for each movie (ignore movies the target user has already rated)
    user_ratings = user_movie_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings.isna()].index
    avg_ratings = similar_user_ratings[unrated_movies].mean(axis=0)
    
    # Filter out movies with an average rating lower than the threshold
    avg_ratings = avg_ratings[avg_ratings > rating_threshold]
    
    # Get user's favorite genres
    favorite_genres = get_favorite_genres(user_id)
    
    # Filter the movies that belong to the user's favorite genres
    recommended_movies = movies[movies['title'].isin(avg_ratings.index)]
    genre_filtered_movies = recommended_movies[recommended_movies[favorite_genres].sum(axis=1) > 0]
    
    # Ensure we only keep the movies present in both avg_ratings and genre_filtered_movies
    valid_recommendations = avg_ratings.index.intersection(genre_filtered_movies['title'])
    
    # Return the top N movie recommendations
    return genre_filtered_movies.set_index('title').loc[valid_recommendations].head(num_recommendations)


# Get movie recommendations for a specific user (e.g., user 196)
recommended_movies = get_recommendations(user_id=196, num_recommendations=5)

# Display the recommended movies
print("\nRecommended Movies for User 196:")
print(recommended_movies)

# Recommend the most popular movies (highest number of ratings)
def recommend_popular_movies(num_recommendations=5):
    popular_movies = merged_data.groupby('title')['rating'].count().sort_values(ascending=False)
    return popular_movies.head(num_recommendations)

# Display the most popular movies
print("\nMost Popular Movies:")
print(recommend_popular_movies())
