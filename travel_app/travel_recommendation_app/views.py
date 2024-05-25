from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
import pickle
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from django.conf import settings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
# Create your views here.

def home(request):
     return render(request, 'index.html')

def incredible(request):
     return render(request,'incredible.html')

def travel_recommendation(request):
     return render(request,'get-recommendation.html')

with open('./static/destination_list.pkl','rb') as file:
    data = pickle.load(file)

with open('./static/similarity_cities.pkl', 'rb') as file:
    model1 = pickle.load(file)

with open('./static/similarity_tags.pkl','rb') as file:
    model2 = pickle.load(file)

cities = data['City'].unique()

def get_city_vector(city):
    return data.get(city)

def load_data(request):
    return render(request, 'cities.html', {'cities': cities})

# Ensure 'Tags' column is preprocessed
data['Tags'] = data['Tags'].apply(lambda x: ' '.join(x.split('|')))

def load_tags(request):
    # Extract unique tags from the DataFrame
    unique_tags = data[['Historical & Heritage', 'Pilgrimage', 'Hill Station', 'Beach', 'Lake & Backwater', 'Wildlife',
                        'Waterfall', 'Nature & Scenic', 'Adventure / Trekking']].columns.tolist()

    # Pass the unique tags to the template
    return render(request, 'tags.html', {'tags': unique_tags})

# Encode categorical variables (Tags)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['Tags'])

# Compute cosine similarity between TF-IDF vectors
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend cities based on a specific city and number of cities
def recommend_city(city, num_cities, cosine_sim=cosine_sim):
    try:
        city_index = data[data['City'] == city].index[0]
    except IndexError:
        return []

    sim_scores = cosine_sim[city_index]
    sim_indices = sim_scores.argsort()[::-1]
    recommended_cities = []

    for idx in sim_indices:
        if idx != city_index:
            recommended_cities.append(data.iloc[idx]['City'])
        if len(recommended_cities) == num_cities:
            break

    return recommended_cities

def recommend_cities(request):
    if request.method == 'POST':
        selected_city = request.POST.get('city')
        number_of_recommendations = int(request.POST.get('number', 1))

        recommendations = recommend_city(selected_city, number_of_recommendations)

        context = {
            'selected_city': selected_city,
            'recommendations': recommendations,
        }
        return render(request, 'recommended_cities.html', context)
    
    return JsonResponse({'error': 'Invalid request method'})

def recommend_cities_based_on_tags(selected_tags, num_cities, cosine_sim=cosine_sim):
    # Compute TF-IDF vector for selected tags
    selected_tags_vector = tfidf.transform([' '.join(selected_tags)])

    # Compute cosine similarity between selected tags vector and all cities
    sim_scores = cosine_similarity(selected_tags_vector, tfidf_matrix).flatten()

    # Get indices of cities with highest similarity scores
    sim_indices = sim_scores.argsort()[::-1]

    # Get top N recommended cities
    recommended_cities = []

    for idx in sim_indices:
        recommended_cities.append(data.iloc[idx]['City'])
        if len(recommended_cities) == num_cities:
            break

    return recommended_cities


def recommend_tags(request):
    if request.method == 'POST':
        selected_tags = request.POST.getlist('tag')
        number_of_recommendations = int(request.POST.get('number', 1))

        recommendations = recommend_cities_based_on_tags(selected_tags, number_of_recommendations)

        # Assuming you have a folder containing city images
        images_folder = os.path.join('./Travel Destinations of India by Rishabh Bafna/')

        # Get a list of tuples where each tuple contains the city name and image path
        city_images = []
        for city in recommendations:
            image_name = city + '.jpg'  # Assuming the image name is the same as the city name
            image_path = os.path.join(images_folder, image_name)
            city_images.append((city, image_path))

        context = {
            'selected_tags': selected_tags,
            'city_images': city_images,
        }
        return render(request, 'recommended_tags.html', context)
    
    return JsonResponse({'error': 'Invalid request method'})

# def text_query_view(request):
#     if request.method == 'POST':
#         text_query = request.POST.get('textQuery')
#         # Process the text query, e.g., retrieve recommendations based on the query
#         # Perform any necessary actions here
#         return HttpResponse("Query submitted successfully!")  # You can customize this response as needed
#     else:
#         return render(request, 'text.html')

 # Update with your dataset path
data = data
places_descriptions = data['description']

# Function to preprocess text data
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    processed_text = ' '.join(tokens)
    return processed_text

# Preprocess the descriptions in the dataset
places_descriptions_processed = places_descriptions.apply(preprocess_text)

# Vectorize the preprocessed text data using TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(places_descriptions_processed)

# Function to recommend places based on user input
def recommend_places(user_input, tfidf_matrix=tfidf_matrix, top_n=25):
    user_input_processed = preprocess_text(user_input)
    user_input_vector = tfidf.transform([user_input_processed])
    similarity_scores = cosine_similarity(user_input_vector, tfidf_matrix)
    top_indices = similarity_scores.argsort()[0][::-1][:top_n]
    recommended_places = data.iloc[top_indices]['City']
    return recommended_places

def recommendation_view(request):
    recommended_places = None
    
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            recommended_cities = recommend_places(user_input)
            image_folder = './static/Travel Destinations of India by Rishabh Bafna/'  # Update with your image folder path
            image_urls = []
            for city in recommended_cities:
                image_filename = city + '.jpg'  # Assuming image filenames match city names
                image_path = os.path.join(image_folder, image_filename)
                image_urls.append((city, image_path))
            recommended_places = image_urls
    
    return render(request, 'recommended_text.html', {'recommended_places': recommended_places})
