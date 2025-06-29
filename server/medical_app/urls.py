from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('download/<int:record_id>/', views.download_markdown, name='download_markdown'),
    path('delete_all/', views.delete_all_summaries, name='delete_all_summaries'),

]
