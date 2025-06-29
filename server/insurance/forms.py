# insurance/forms.py
from django import forms
from .models import InsuranceDocument, INSURANCE_TYPES

class DocumentUploadForm(forms.Form):
    """Form for uploading insurance documents"""
    title = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter document title (e.g., Car Insurance Policies 2024)'
        })
    )
    document = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.pdf',
            'title': 'Upload PDF document'
        })
    )
    insurance_type = forms.ChoiceField(
        choices=INSURANCE_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Select the type of insurance document.'
    )

    def clean_document(self):
        document = self.cleaned_data['document']
        if document.size > 50 * 1024 * 1024:
            raise forms.ValidationError('File size must be less than 50MB.')
        if not document.name.lower().endswith('.pdf'):
            raise forms.ValidationError('Only PDF files are allowed.')
        return document

class InsuranceQueryForm(forms.Form):
    """Form for insurance queries"""
    QUERY_PLACEHOLDERS = {
        'health': 'Enter your health insurance query (e.g., "I am 30 years old with diabetes, looking for health insurance under ₹50,000 per year")',
        'car': 'Enter your car insurance query (e.g., "I have a 2019 Honda City, looking for comprehensive insurance under ₹20,000 per year")',
        'life': 'Enter your life insurance query (e.g., "I am 35 years old, looking for a term plan with ₹1 crore coverage")',
        'home': 'Enter your home insurance query (e.g., "I own a 2BHK in Mumbai, looking for home insurance covering natural disasters")',
    }

    query = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
        }),
        help_text='Be specific about your requirements for better recommendations.'
    )
    insurance_type = forms.ChoiceField(
        choices=INSURANCE_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Select the type of insurance query.'
    )

    def __init__(self, *args, **kwargs):
        insurance_type = kwargs.pop('insurance_type', 'health')
        super().__init__(*args, **kwargs)
        self.fields['query'].widget.attrs['placeholder'] = self.QUERY_PLACEHOLDERS.get(insurance_type, 'Enter your insurance query')

    def clean_query(self):
        query = self.cleaned_data['query']
        if len(query.strip()) < 10:
            raise forms.ValidationError('Query must be at least 10 characters long.')
        if len(query) > 10000:
            raise forms.ValidationError('Query must be less than 10000 characters.')
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
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    insurance_type = forms.ChoiceField(
        choices=INSURANCE_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Select the type of insurance document to search.'
    )