# insurance/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
from django.views import View
import json
import os
import logging
from pathlib import Path
from .models import InsuranceDocument, DocumentChunk, InsuranceQuery, INSURANCE_TYPES
from .forms import DocumentUploadForm, InsuranceQueryForm
from ai_modules.insurance_processor import InsuranceRAGProcessor

from django.views import View
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from .models import InsuranceDocument, DocumentChunk, InsuranceQuery, INSURANCE_TYPES
from .forms import DocumentUploadForm, InsuranceQueryForm
from ai_modules.insurance_processor import InsuranceRAGProcessor
import logging
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

logger = logging.getLogger(__name__)

class InsuranceIndexView(View):
    def get(self, request, insurance_type):
        if insurance_type not in dict(INSURANCE_TYPES):
            messages.error(request, 'Invalid insurance type.')
            return redirect('insurance:index', insurance_type='health')
        
        documents = InsuranceDocument.objects.filter(insurance_type=insurance_type)
        recent_queries = InsuranceQuery.objects.filter(insurance_type=insurance_type).order_by('-query_time')[:5]
        
        context = {
            'documents': documents,
            'recent_queries': recent_queries,
            'upload_form': DocumentUploadForm(initial={'insurance_type': insurance_type}, insurance_type=insurance_type),
            'query_form': InsuranceQueryForm(initial={
                'query': request.session.get('insurance_query', ''),
                'insurance_type': insurance_type
            }, insurance_type=insurance_type),
            'total_chunks': DocumentChunk.objects.filter(insurance_type=insurance_type).count(),
            'total_documents': documents.count(),
            'insurance_type': insurance_type,
        }
        return render(request, 'insurance/index.html', context)

class DocumentUploadView(View):
    """Handle document upload and processing"""
    def post(self, request, insurance_type):
        if insurance_type not in dict(INSURANCE_TYPES):
            messages.error(request, 'Invalid insurance type.')
            return redirect('insurance:index', insurance_type='health')
        
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploaded_file = request.FILES['document']
                title = form.cleaned_data['title']
                form_insurance_type = form.cleaned_data['insurance_type']
                
                if form_insurance_type != insurance_type:
                    messages.error(request, 'Insurance type mismatch.')
                    return redirect('insurance:index', insurance_type=insurance_type)
                
                if not uploaded_file.name.lower().endswith('.pdf'):
                    messages.error(request, 'Only PDF files are allowed.')
                    return redirect('insurance:index', insurance_type=insurance_type)
                
                file_name = f"insurance_docs/{insurance_type}/{uploaded_file.name}"
                file_path = default_storage.save(file_name, ContentFile(uploaded_file.read()))
                full_file_path = default_storage.path(file_path)
                
                document = InsuranceDocument.objects.create(
                    title=title,
                    file_path=full_file_path,
                    original_filename=uploaded_file.name,
                    insurance_type=insurance_type
                )
                
                processor = InsuranceRAGProcessor(insurance_type=insurance_type)
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
        
        return redirect('insurance:index', insurance_type=insurance_type)

# insurance/views.py (update InsuranceQueryView.post)
class InsuranceQueryView(View):
    def get(self, request, insurance_type):
        if insurance_type not in dict(INSURANCE_TYPES):
            messages.error(request, 'Invalid insurance type.')
            return redirect('insurance:index', insurance_type='health')
        
        query_text = request.session.pop('insurance_query', None)
        form = InsuranceQueryForm(
            initial={'query': query_text, 'insurance_type': insurance_type},
            insurance_type=insurance_type
        )
        
        documents = InsuranceDocument.objects.filter(insurance_type=insurance_type)
        recent_queries = InsuranceQuery.objects.filter(insurance_type=insurance_type).order_by('-query_time')[:5]
        
        context = {
            'documents': documents,
            'recent_queries': recent_queries,
            'upload_form': DocumentUploadForm(initial={'insurance_type': insurance_type}, insurance_type=insurance_type),
            'query_form': form,
            'total_chunks': DocumentChunk.objects.filter(insurance_type=insurance_type).count(),
            'total_documents': documents.count(),
            'insurance_type': insurance_type,
        }
        return render(request, 'insurance/index.html', context)

    def post(self, request, insurance_type):
        if insurance_type not in dict(INSURANCE_TYPES):
            return JsonResponse({'success': False, 'error': 'Invalid insurance type.'})
        
        form = InsuranceQueryForm(request.POST, insurance_type=insurance_type)
        if form.is_valid():
            try:
                query_text = form.cleaned_data['query']
                form_insurance_type = form.cleaned_data['insurance_type']
                
                if form_insurance_type != insurance_type:
                    return JsonResponse({'success': False, 'error': 'Insurance type mismatch.'})
                
                if not DocumentChunk.objects.filter(insurance_type=insurance_type).exists():
                    return JsonResponse({
                        'success': False,
                        'error': f'No processed {insurance_type} documents found. Please upload and process documents first.'
                    })
                
                processor = InsuranceRAGProcessor(insurance_type=insurance_type)
                success = processor.initialize_system()
                
                if not success:
                    return JsonResponse({
                        'success': False,
                        'error': f'Failed to initialize {insurance_type} RAG system.'
                    })
                
                response = processor.query_insurance(query_text)
                
                return JsonResponse({
                    'success': True,
                    'response': response,
                    'query': query_text
                })
            except Exception as e:
                logger.error(f"Error processing {insurance_type} query: {e}")
                return JsonResponse({
                    'success': False,
                    'error': f'Error processing {insurance_type} query: {str(e)}'
                })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Invalid query form: ' + str(form.errors)
            })
    """Handle insurance queries"""
    def get(self, request, insurance_type):
        if insurance_type not in dict(INSURANCE_TYPES):
            messages.error(request, 'Invalid insurance type.')
            return redirect('insurance:index', insurance_type='health')
        
        query_text = request.session.pop('insurance_query', None)
        form = InsuranceQueryForm(initial={'query': query_text, 'insurance_type': insurance_type}, insurance_type=insurance_type)
        
        documents = InsuranceDocument.objects.filter(insurance_type=insurance_type)
        recent_queries = InsuranceQuery.objects.filter(insurance_type=insurance_type)[:5]
        
        context = {
            'documents': documents,
            'recent_queries': recent_queries,
            'upload_form': DocumentUploadForm(initial={'insurance_type': insurance_type}),
            'query_form': form,
            'total_chunks': DocumentChunk.objects.filter(insurance_type=insurance_type).count(),
            'total_documents': documents.count(),
            'insurance_type': insurance_type,
        }
        return render(request, 'insurance/index.html', context)

    def post(self, request, insurance_type):
        if insurance_type not in dict(INSURANCE_TYPES):
            return JsonResponse({'success': False, 'error': 'Invalid insurance type.'})
        
        form = InsuranceQueryForm(request.POST, insurance_type=insurance_type)
        if form.is_valid():
            try:
                query_text = form.cleaned_data['query']
                form_insurance_type = form.cleaned_data['insurance_type']
                
                if form_insurance_type != insurance_type:
                    return JsonResponse({'success': False, 'error': 'Insurance type mismatch.'})
                
                if not DocumentChunk.objects.filter(insurance_type=insurance_type).exists():
                    return JsonResponse({
                        'success': False,
                        'error': f'No processed {insurance_type} documents found. Please upload and process documents first.'
                    })
                
                processor = InsuranceRAGProcessor(insurance_type=insurance_type)
                success = processor.initialize_system()
                
                if not success:
                    return JsonResponse({
                        'success': False,
                        'error': f'Failed to initialize {insurance_type} RAG system. Please ensure the system is properly configured.'
                    })
                
                response = processor.query_insurance(query_text)
                
                return JsonResponse({
                    'success': True,
                    'response': response,
                    'query': query_text
                })
            except Exception as e:
                logger.error(f"Error processing {insurance_type} query: {e}")
                return JsonResponse({
                    'success': False,
                    'error': f'Error processing {insurance_type} query: {str(e)}'
                })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Invalid query form: ' + str(form.errors)
            })
@csrf_exempt
def clear_database(request, insurance_type):
    """Clear all data for a specific insurance type"""
    if insurance_type not in dict(INSURANCE_TYPES):
        messages.error(request, 'Invalid insurance type.')
        return redirect('insurance:index', insurance_type='health')
    
    if request.method == 'POST':
        try:
            success = InsuranceRAGProcessor.clear_all_data(insurance_type=insurance_type)
            if success:
                messages.success(request, f'All {insurance_type} data cleared successfully!')
            else:
                messages.error(request, f'Failed to clear {insurance_type} data.')
        except Exception as e:
            logger.error(f"Error clearing {insurance_type} database: {e}")
            messages.error(request, f'Error clearing {insurance_type} database: {str(e)}')
    
    return redirect('insurance:index', insurance_type=insurance_type)

def document_detail(request, insurance_type, document_id):
    """Show document details and chunks"""
    if insurance_type not in dict(INSURANCE_TYPES):
        messages.error(request, 'Invalid insurance type.')
        return redirect('insurance:index', insurance_type='health')
    
    document = get_object_or_404(InsuranceDocument, id=document_id, insurance_type=insurance_type)
    chunks = DocumentChunk.objects.filter(document=document, insurance_type=insurance_type)
    
    context = {
        'document': document,
        'chunks': chunks,
        'chunk_strategies': {
            'policy': chunks.filter(strategy='policy').count(),
            'semantic': chunks.filter(strategy='semantic').count(),
            'header': chunks.filter(strategy='header').count(),
        },
        'insurance_type': insurance_type,
    }
    return render(request, 'insurance/document_detail.html', context)

def delete_document(request, insurance_type, document_id):
    """Delete a document and its chunks"""
    if insurance_type not in dict(INSURANCE_TYPES):
        messages.error(request, 'Invalid insurance type.')
        return redirect('insurance:index', insurance_type='health')
    
    if request.method == 'POST':
        document = get_object_or_404(InsuranceDocument, id=document_id, insurance_type=insurance_type)
        try:
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
            document_title = document.title
            document.delete()
            messages.success(request, f'Document "{document_title}" deleted successfully!')
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            messages.error(request, f'Error deleting document: {str(e)}')
    
    return redirect('insurance:index', insurance_type=insurance_type)

def query_history(request, insurance_type):
    """Show query history"""
    if insurance_type not in dict(INSURANCE_TYPES):
        messages.error(request, 'Invalid insurance type.')
        return redirect('insurance:index', insurance_type='health')
    
    queries = InsuranceQuery.objects.filter(insurance_type=insurance_type)
    context = {
        'queries': queries,
        'insurance_type': insurance_type,
    }
    return render(request, 'insurance/query_history.html', context)

def reprocess_document(request, insurance_type, document_id):
    """Reprocess a document"""
    if insurance_type not in dict(INSURANCE_TYPES):
        messages.error(request, 'Invalid insurance type.')
        return redirect('insurance:index', insurance_type='health')
    
    if request.method == 'POST':
        document = get_object_or_404(InsuranceDocument, id=document_id, insurance_type=insurance_type)
        try:
            processor = InsuranceRAGProcessor(insurance_type=insurance_type)
            success = processor.process_document(document.file_path, document.id)
            if success:
                messages.success(request, f'Document "{document.title}" reprocessed successfully!')
            else:
                messages.error(request, 'Failed to reprocess document.')
        except Exception as e:
            logger.error(f"Error reprocessing document: {e}")
            messages.error(request, f'Error reprocessing document: {str(e)}')
    
    return redirect('insurance:document_detail', insurance_type=insurance_type, document_id=document_id)

def export_chunks(request, insurance_type, document_id):
    """Export document chunks as JSON"""
    if insurance_type not in dict(INSURANCE_TYPES):
        messages.error(request, 'Invalid insurance type.')
        return redirect('insurance:index', insurance_type='health')
    
    document = get_object_or_404(InsuranceDocument, id=document_id, insurance_type=insurance_type)
    chunks = DocumentChunk.objects.filter(document=document, insurance_type=insurance_type)
    
    chunks_data = []
    for chunk in chunks:
        chunks_data.append({
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'strategy': chunk.strategy,
            'metadata': chunk.metadata,
            'created_at': chunk.created_at.isoformat(),
            'insurance_type': chunk.insurance_type,
        })
    
    response = HttpResponse(
        json.dumps(chunks_data, indent=2),
        content_type='application/json'
    )
    response['Content-Disposition'] = f'attachment; filename="{document.title}_chunks.json"'
    return response


def system_status(request, insurance_type):
    """Show system status and statistics"""
    if insurance_type not in dict(INSURANCE_TYPES):
        messages.error(request, 'Invalid insurance type.')
        return redirect('insurance:index', insurance_type='health')
    
    try:
        processor = InsuranceRAGProcessor(insurance_type=insurance_type)
        ollama_status = True
        try:
            test_embedding = processor.embeddings.embed_query("test")
            ollama_status = len(test_embedding) > 0
        except:
            ollama_status = False
        
        context = {
            'ollama_status': ollama_status,
            'total_documents': InsuranceDocument.objects.filter(insurance_type=insurance_type).count(),
            'processed_documents': InsuranceDocument.objects.filter(insurance_type=insurance_type, processed=True).count(),
            'total_chunks': DocumentChunk.objects.filter(insurance_type=insurance_type).count(),
            'total_queries': InsuranceQuery.objects.filter(insurance_type=insurance_type).count(),
            'chunk_strategies': {
                'policy': DocumentChunk.objects.filter(insurance_type=insurance_type, strategy='policy').count(),
                'semantic': DocumentChunk.objects.filter(insurance_type=insurance_type, strategy='semantic').count(),
                'header': DocumentChunk.objects.filter(insurance_type=insurance_type, strategy='header').count(),
            },
            'recent_queries': InsuranceQuery.objects.filter(insurance_type=insurance_type).order_by('-query_time')[:10],
            'insurance_type': insurance_type,
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        context = {
            'error': str(e),
            'ollama_status': False,
            'insurance_type': insurance_type,
        }
    
    return render(request, 'insurance/system_status.html', context)