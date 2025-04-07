from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np
import pickle
import ast
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

api = Blueprint('api', __name__)

# Load model
model_path = os.path.join(os.path.dirname(_file_), 'models', 'recipe_recommendation_model.pkl')
with open(model_path, 'rb') as f:
    vocab = pickle.load(f)

# Load processed CSV
csv_path = os.path.join(os.path.dirname(_file_), 'data', 'processed', 'recipes_processed.csv')
df = pd.read_csv(csv_path)
df['Ingredient_Vector'] = df['Ingredient_Vector'].apply(ast.literal_eval)

# Vectorize function
def vectorize_ingredients(ingredient_str, vocab):
    vector = [0] * len(vocab)
    ingredients = [ingredient.strip().lower() for ingredient in re.split(r',|\s', ingredient_str) if ingredient]
    for ingredient in ingredients:
        if ingredient in vocab:
            vector[vocab.index(ingredient)] = 1
    return vector

# Recommendation function
def recommend_recipes(user_vector, df, top_n=5):
    recipe_vectors = np.array(df['Ingredient_Vector'].tolist())
    user_vector = vectorize_ingredients(user_vector, vocab)
    user_vector = np.array(user_vector).reshape(1, -1)
    similarity_scores = cosine_similarity(user_vector, recipe_vectors)[0]
    
    recipe_similarity = list(enumerate(similarity_scores))
    top_indices = sorted(recipe_similarity, key=lambda x: x[1], reverse=True)[:top_n]
    top_recipe_indices = [index for index, score in top_indices]
    
    return top_recipe_indices

# API route for recommendation
@api.route('/recommend', methods=['POST', 'OPTIONS'])
def recommend():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OK'}), 200

    data = request.get_json()
    ingredients = data.get("ingredients", [])
    
    # Convert list back to comma string for vectorization
    user_input = ', '.join(ingredients)

    top_recipes = recommend_recipes(user_input, df, top_n=5)

    results = []
    for recipe in top_recipes:
        name = df.loc[recipe, 'Title']
        ingredients = df.loc[recipe, 'Core_Ingredients']
        image_url = df.loc[recipe, 'Image Link']
        formatted_ingredients = ', '.join(ingredients.split())

        results.append({
            'name': name,
            'ingredients': formatted_ingredients,
            'image_url': image_url
        })

    return jsonify(results)