# insurance/forms.py
from django import forms
from .models import InsuranceDocument

class DocumentUploadForm(forms.Form):
    """Form for uploading insurance documents"""
    title = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter document title (e.g., Health Insurance Policies 2024)'
        })
    )
    
    document = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.pdf',
            'title': 'Upload PDF document'
        })
    )
    
    def clean_document(self):
        document = self.cleaned_data['document']
        
        # Check file size (max 50MB)
        if document.size > 50 * 1024 * 1024:
            raise forms.ValidationError('File size must be less than 50MB.')
        
        # Check file extension
        if not document.name.lower().endswith('.pdf'):
            raise forms.ValidationError('Only PDF files are allowed.')
        
        return document

class InsuranceQueryForm(forms.Form):
    """Form for insurance queries"""
    query = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'Enter your insurance query (e.g., "I am 30 years old with diabetes, looking for health insurance under ₹50,000 per year")'
        }),
        help_text='Be specific about your age, health conditions, budget, and preferences for better recommendations.'
    )
    
    def clean_query(self):
        query = self.cleaned_data['query']
        
        # Check minimum length
        if len(query.strip()) < 10:
            raise forms.ValidationError('Query must be at least 10 characters long.')
        
        # Check maximum length
        if len(query) > 1000:
            raise forms.ValidationError('Query must be less than 1000 characters.')
        
        return query.strip()

class DocumentSearchForm(forms.Form):
    """Form for searching documents"""
    search_term = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search documents...'
        })
    )
    
    strategy = forms.ChoiceField(
        choices=[
            ('', 'All Strategies'),
            ('policy', 'Policy-based'),
            ('semantic', 'Semantic'),
            ('header', 'Header-based'),
        ],
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )