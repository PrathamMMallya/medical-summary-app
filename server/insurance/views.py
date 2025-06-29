# insurance/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
import os
import logging
from pathlib import Path

from .models import InsuranceDocument, DocumentChunk, InsuranceQuery
from .forms import DocumentUploadForm, InsuranceQueryForm

# Import the RAG processor
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai_modules'))
from ai_modules.insurance_processor import InsuranceRAGProcessor

logger = logging.getLogger(__name__)

class InsuranceIndexView(View):
    """Main insurance app view"""
    
    def get(self, request):
        documents = InsuranceDocument.objects.all()
        recent_queries = InsuranceQuery.objects.all()[:5]
        
        context = {
            'documents': documents,
            'recent_queries': recent_queries,
            'upload_form': DocumentUploadForm(),
            'query_form': InsuranceQueryForm(),
            'total_chunks': DocumentChunk.objects.count(),
            'total_documents': documents.count(),
        }
        return render(request, 'insurance/index.html', context)

class DocumentUploadView(View):
    """Handle document upload and processing"""
    
    def post(self, request):
        form = DocumentUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                uploaded_file = request.FILES['document']
                title = form.cleaned_data['title']
                
                # Validate file type
                if not uploaded_file.name.lower().endswith('.pdf'):
                    messages.error(request, 'Only PDF files are allowed.')
                    return redirect('insurance:index')
                
                # Save file
                file_name = f"insurance_docs/{uploaded_file.name}"
                file_path = default_storage.save(file_name, ContentFile(uploaded_file.read()))
                full_file_path = default_storage.path(file_path)
                
                # Create document record
                document = InsuranceDocument.objects.create(
                    title=title,
                    file_path=full_file_path,
                    original_filename=uploaded_file.name
                )
                
                # Process document in background (you might want to use Celery for this)
                processor = InsuranceRAGProcessor()
                success = processor.process_document(full_file_path, document.id)
                
                if success:
                    messages.success(request, f'Document "{title}" uploaded and processed successfully!')
                else:
                    messages.error(request, 'Document uploaded but processing failed.')
                    
            except Exception as e:
                logger.error(f"Error uploading document: {e}")
                messages.error(request, f'Error uploading document: {str(e)}')
        else:
            messages.error(request, 'Invalid form data.')
        
        return redirect('insurance:index')

class InsuranceQueryView(View):
    """Handle insurance queries (GET shows summary from session if available)"""

    def get(self, request):
        query_text = request.session.get('insurance_query', '')
        initial_form = InsuranceQueryForm(initial={'query': query_text}) if query_text else InsuranceQueryForm()

        documents = InsuranceDocument.objects.all()
        recent_queries = InsuranceQuery.objects.all()[:5]

        context = {
            'documents': documents,
            'recent_queries': recent_queries,
            'upload_form': DocumentUploadForm(),
            'query_form': initial_form,
            'insurance_summary': query_text,
            'total_chunks': DocumentChunk.objects.count(),
            'total_documents': documents.count(),
        }

        return render(request, 'insurance/index.html', context)
    def post(self, request):
        form = InsuranceQueryForm(request.POST)

        if form.is_valid():
            try:
                query_text = form.cleaned_data['query']

                if not DocumentChunk.objects.exists():
                    return JsonResponse({
                        'success': False,
                        'error': 'No processed documents found. Please upload and process documents first.'
                    })

                processor = InsuranceRAGProcessor()
                success = processor.initialize_system()

                if not success:
                    return JsonResponse({
                        'success': False,
                        'error': 'Failed to initialize RAG system.'
                    })

                response = processor.query_insurance(query_text)

                return JsonResponse({
                    'success': True,
                    'response': response,
                    'query': query_text
                })

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                return JsonResponse({
                    'success': False,
                    'error': f'Error processing query: {str(e)}'
                })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Invalid query form: ' + str(form.errors)
            })


@csrf_exempt
def clear_database(request):
    """Clear all data from database"""
    if request.method == 'POST':
        try:
            success = InsuranceRAGProcessor.clear_all_data()
            if success:
                messages.success(request, 'All data cleared successfully!')
            else:
                messages.error(request, 'Failed to clear data.')
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            messages.error(request, f'Error clearing database: {str(e)}')
    
    return redirect('insurance:index')

def document_detail(request, document_id):
    """Show document details and chunks"""
    document = get_object_or_404(InsuranceDocument, id=document_id)
    chunks = DocumentChunk.objects.filter(document=document)
    
    context = {
        'document': document,
        'chunks': chunks,
        'chunk_strategies': {
            'policy': chunks.filter(strategy='policy').count(),
            'semantic': chunks.filter(strategy='semantic').count(),
            'header': chunks.filter(strategy='header').count(),
        }
    }
    return render(request, 'insurance/document_detail.html', context)

def delete_document(request, document_id):
    """Delete a document and its chunks"""
    if request.method == 'POST':
        document = get_object_or_404(InsuranceDocument, id=document_id)
        
        try:
            # Delete file from storage
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
            
            # Delete document (chunks will be deleted due to CASCADE)
            document_title = document.title
            document.delete()
            
            messages.success(request, f'Document "{document_title}" deleted successfully!')
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            messages.error(request, f'Error deleting document: {str(e)}')
    
    return redirect('insurance:index')

def query_history(request):
    """Show query history"""
    queries = InsuranceQuery.objects.all()
    context = {
        'queries': queries
    }
    return render(request, 'insurance/query_history.html', context)

def reprocess_document(request, document_id):
    """Reprocess a document"""
    if request.method == 'POST':
        document = get_object_or_404(InsuranceDocument, id=document_id)
        
        try:
            processor = InsuranceRAGProcessor()
            success = processor.process_document(document.file_path, document.id)
            
            if success:
                messages.success(request, f'Document "{document.title}" reprocessed successfully!')
            else:
                messages.error(request, 'Failed to reprocess document.')
                
        except Exception as e:
            logger.error(f"Error reprocessing document: {e}")
            messages.error(request, f'Error reprocessing document: {str(e)}')
    
    return redirect('insurance:document_detail', document_id=document_id)

def export_chunks(request, document_id):
    """Export document chunks as JSON"""
    document = get_object_or_404(InsuranceDocument, id=document_id)
    chunks = DocumentChunk.objects.filter(document=document)
    
    chunks_data = []
    for chunk in chunks:
        chunks_data.append({
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'strategy': chunk.strategy,
            'metadata': chunk.metadata,
            'created_at': chunk.created_at.isoformat()
        })
    
    response = HttpResponse(
        json.dumps(chunks_data, indent=2),
        content_type='application/json'
    )
    response['Content-Disposition'] = f'attachment; filename="{document.title}_chunks.json"'
    return response

def system_status(request):
    """Show system status and statistics"""
    try:
        # Test Ollama connection
        processor = InsuranceRAGProcessor()
        ollama_status = True
        try:
            test_embedding = processor.embeddings.embed_query("test")
            ollama_status = len(test_embedding) > 0
        except:
            ollama_status = False
        
        context = {
            'ollama_status': ollama_status,
            'total_documents': InsuranceDocument.objects.count(),
            'processed_documents': InsuranceDocument.objects.filter(processed=True).count(),
            'total_chunks': DocumentChunk.objects.count(),
            'total_queries': InsuranceQuery.objects.count(),
            'chunk_strategies': {
                'policy': DocumentChunk.objects.filter(strategy='policy').count(),
                'semantic': DocumentChunk.objects.filter(strategy='semantic').count(),
                'header': DocumentChunk.objects.filter(strategy='header').count(),
            },
            'recent_queries': InsuranceQuery.objects.order_by('-query_time')[:10],
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        context = {
            'error': str(e),
            'ollama_status': False,
        }
    
    return render(request, 'insurance/system_status.html', context)