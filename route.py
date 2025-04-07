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
model_path = os.path.join(os.path.dirname(__file__), 'models', 'recipe_recommendation_model.pkl')
with open(model_path, 'rb') as f:
    vocab = pickle.load(f)

# Load processed CSV
csv_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'recipes_processed.csv')
df = pd.read_csv(csv_path)
df['Ingredient_Vector'] = df['Ingredient_Vector'].apply(ast.literal_eval)

# Vectorize function
def vectorize_ingredients(ingredient_str, vocab):
    vector = [0] * len(vocab)
    ingredients = [ingredient.strip().lower() for ingredient in re.split(r',|\s+', ingredient_str) if ingredient]
    for ingredient in ingredients:
        if ingredient in vocab:
            vector[vocab.index(ingredient)] = 1
    return vector

# Recommendation function
def recommend_recipes(user_input, df, top_n=5):
    user_vector = vectorize_ingredients(user_input, vocab)
    user_vector = np.array(user_vector).reshape(1, -1)
    recipe_vectors = np.array(df['Ingredient_Vector'].tolist())
    similarity_scores = cosine_similarity(user_vector, recipe_vectors)[0]
    
    top_indices = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)[:top_n]
    return [index for index, score in top_indices]

# API route for recommendation
@api.route('/recommend', methods=['POST', 'OPTIONS'])
def recommend():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'CORS preflight OK'}), 200

    data = request.get_json()
    ingredients_list = data.get("ingredients", [])

    if not ingredients_list or not isinstance(ingredients_list, list):
        return jsonify({"error": "Invalid ingredients list"}), 400

    user_input = ', '.join(ingredients_list)
    top_recipes = recommend_recipes(user_input, df, top_n=5)

    results = []
    for index in top_recipes:
        name = df.loc[index, 'Title']
        core_ingredients = df.loc[index, 'Core_Ingredients']
        image_url = df.loc[index, 'Image Link']
        formatted_ingredients = ', '.join(core_ingredients.split())

        results.append({
            'name': name,
            'ingredients': formatted_ingredients,
            'image_url': image_url
        })

    return jsonify(results)
