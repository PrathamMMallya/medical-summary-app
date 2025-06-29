# insurance/urls.py
from django.urls import path
from . import views
from .views import InsuranceQueryView

app_name = 'insurance'

urlpatterns = [
    # Main views
    path('', views.InsuranceIndexView.as_view(), name='index'),
    
    # Document management
    path('upload/', views.DocumentUploadView.as_view(), name='upload_document'),
    path('document/<int:document_id>/', views.document_detail, name='document_detail'),
    path('document/<int:document_id>/delete/', views.delete_document, name='delete_document'),
    path('document/<int:document_id>/reprocess/', views.reprocess_document, name='reprocess_document'),
    path('document/<int:document_id>/export/', views.export_chunks, name='export_chunks'),
    
    # Query management
    path('query/', views.InsuranceQueryView.as_view(), name='query'),
    path('query-history/', views.query_history, name='query_history'),
    
    # System management
    path('clear-database/', views.clear_database, name='clear_database'),
    path('system-status/', views.system_status, name='system_status'),
]