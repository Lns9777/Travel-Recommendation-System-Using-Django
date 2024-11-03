from django.urls import path
from travel_recommendation_app import views

urlpatterns = [
    path('', views.home, name='travel_recommedation_app'),
    path('incredible.html',views.incredible,name='incredible-india'),
    path('get-recommendation.html',views.travel_recommendation,name='Travel Recommendation'),
    path('cities.html',views.load_data,name='City'),
    path('tags.html',views.load_tags,name='Tags'),
    # path('text.html',views.text_query_view,name='Text'),
    path('recommended_cities.html',views.recommend_cities,name='city_recommended'),
    path('recommended_tags.html',views.recommend_tags,name='tags_recommended'),
    path('recommended_text.html',views.recommendation_view,name='Text_recommended'),
    path('booking.html',views.booking_form,name='booking_form'),
]
