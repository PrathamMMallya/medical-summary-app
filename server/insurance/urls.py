# insurance/urls.py
from django.urls import path
from . import views
from .views import InsuranceIndexView, DocumentUploadView, InsuranceQueryView

app_name = 'insurance'

urlpatterns = [
    # Main views for each insurance type
    path('<str:insurance_type>/', views.InsuranceIndexView.as_view(), name='index'),
    # Document management
    path('<str:insurance_type>/upload/', views.DocumentUploadView.as_view(), name='upload_document'),
    path('<str:insurance_type>/document/<int:document_id>/', views.document_detail, name='document_detail'),
    path('<str:insurance_type>/document/<int:document_id>/delete/', views.delete_document, name='delete_document'),
    path('<str:insurance_type>/document/<int:document_id>/reprocess/', views.reprocess_document, name='reprocess_document'),
    path('<str:insurance_type>/document/<int:document_id>/export/', views.export_chunks, name='export_chunks'),
    # Query management
    path('<str:insurance_type>/query/', views.InsuranceQueryView.as_view(), name='query'),
    path('<str:insurance_type>/query-history/', views.query_history, name='query_history'),
    # System management
    path('<str:insurance_type>/clear-database/', views.clear_database, name='clear_database'),
    path('<str:insurance_type>/system-status/', views.system_status, name='system_status'),
]