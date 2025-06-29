# insurance/models.py
from django.db import models
from django.contrib.auth.models import User
import json

class InsuranceDocument(models.Model):
    """Model to store uploaded insurance documents"""
    title = models.CharField(max_length=200)
    file_path = models.CharField(max_length=500)
    original_filename = models.CharField(max_length=200)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    total_chunks = models.IntegerField(default=0)
    
    def __str__(self):
        return f"{self.title} - {self.original_filename}"
    
    class Meta:
        ordering = ['-uploaded_at']

class DocumentChunk(models.Model):
    """Model to store document chunks for RAG"""
    STRATEGY_CHOICES = [
        ('policy', 'Policy-based'),
        ('semantic', 'Semantic'),
        ('header', 'Header-based'),
    ]
    
    document = models.ForeignKey(InsuranceDocument, on_delete=models.CASCADE, related_name='chunks')
    chunk_id = models.CharField(max_length=100)  # Unique identifier for the chunk
    content = models.TextField()
    strategy = models.CharField(max_length=20, choices=STRATEGY_CHOICES)
    metadata = models.JSONField(default=dict)  # Store additional metadata
    embedding_vector = models.JSONField(null=True, blank=True)  # Store embeddings as JSON
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Chunk {self.chunk_id} - {self.document.title}"
    
    def set_embedding(self, embedding_list):
        """Store embedding vector as JSON"""
        self.embedding_vector = embedding_list
    
    def get_embedding(self):
        """Retrieve embedding vector"""
        return self.embedding_vector
    
    class Meta:
        ordering = ['chunk_id']
        unique_together = ['document', 'chunk_id']

class InsuranceQuery(models.Model):
    """Model to store user queries and responses"""
    query_text = models.TextField()
    response_text = models.TextField()
    retrieved_chunks = models.JSONField(default=list)  # Store chunk IDs that were used
    query_time = models.DateTimeField(auto_now_add=True)
    processing_time = models.FloatField(null=True, blank=True)  # Time taken to process
    
    def __str__(self):
        return f"Query: {self.query_text[:50]}..."
    
    def set_retrieved_chunks(self, chunk_ids):
        """Store the IDs of chunks that were retrieved for this query"""
        self.retrieved_chunks = chunk_ids
    
    class Meta:
        ordering = ['-query_time']