# insurance/models.py
from django.db import models
from django.contrib.auth.models import User

INSURANCE_TYPES = [
    ('health', 'Health Insurance'),
    ('car', 'Car Insurance'),
    ('life', 'Life Insurance'),
    ('home', 'Home Insurance'),
]

class InsuranceDocument(models.Model):
    """Model to store uploaded insurance documents"""
    title = models.CharField(max_length=200)
    file_path = models.CharField(max_length=500)
    original_filename = models.CharField(max_length=200)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    total_chunks = models.IntegerField(default=0)
    insurance_type = models.CharField(max_length=20, choices=INSURANCE_TYPES, default='health')

    def __str__(self):
        return f"{self.title} - {self.original_filename} ({self.insurance_type})"

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
    chunk_id = models.CharField(max_length=100)
    content = models.TextField()
    strategy = models.CharField(max_length=20, choices=STRATEGY_CHOICES)
    metadata = models.JSONField(default=dict)
    embedding_vector = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    insurance_type = models.CharField(max_length=20, choices=INSURANCE_TYPES, default='health')

    def __str__(self):
        return f"Chunk {self.chunk_id} - {self.document.title} ({self.insurance_type})"

    def set_embedding(self, embedding_list):
        self.embedding_vector = embedding_list

    def get_embedding(self):
        return self.embedding_vector

    class Meta:
        ordering = ['chunk_id']
        unique_together = ['document', 'chunk_id']

class InsuranceQuery(models.Model):
    """Model to store user queries and responses"""
    query_text = models.TextField()
    response_text = models.TextField()
    retrieved_chunks = models.JSONField(default=list)
    query_time = models.DateTimeField(auto_now_add=True)
    processing_time = models.FloatField(null=True, blank=True)
    insurance_type = models.CharField(max_length=20, choices=INSURANCE_TYPES, default='health')

    def __str__(self):
        return f"Query: {self.query_text[:50]}... ({self.insurance_type})"

    def set_retrieved_chunks(self, chunk_ids):
        self.retrieved_chunks = chunk_ids

    class Meta:
        ordering = ['-query_time']