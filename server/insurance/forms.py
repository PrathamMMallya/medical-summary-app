# insurance/forms.py
from django import forms
from .models import InsuranceDocument, INSURANCE_TYPES
# insurance/forms.py
from django import forms
from .models import INSURANCE_TYPES

QUERY_PLACEHOLDERS = {
    'health': 'Enter your health insurance query (e.g., "I am 30 years old with diabetes, looking for health insurance under ₹50,000 per year")',
    'car': 'Enter your car insurance query (e.g., "I have a 2019 Honda City, looking for comprehensive insurance under ₹20,000 per year")',
    'life': 'Enter your life insurance query (e.g., "I am 35 years old, looking for a term plan with ₹1 crore coverage")',
    'home': 'Enter your home insurance query (e.g., "I own a 2BHK in Mumbai, looking for home insurance covering natural disasters")',
}

class DocumentUploadForm(forms.Form):
    title = forms.CharField(max_length=200, label="Document Title")
    document = forms.FileField(label="PDF File")
    insurance_type = forms.ChoiceField(choices=INSURANCE_TYPES, label="Insurance Type")

    def __init__(self, *args, insurance_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        if insurance_type:
            self.fields['insurance_type'].initial = insurance_type
            self.fields['insurance_type'].widget.attrs['readonly'] = True

class InsuranceQueryForm(forms.Form):
    query = forms.CharField(widget=forms.Textarea(attrs={'rows': 4}), label="Your Question")
    insurance_type = forms.ChoiceField(choices=INSURANCE_TYPES, label="Insurance Type")

    def __init__(self, *args, insurance_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        if insurance_type:
            self.fields['insurance_type'].initial = insurance_type
            self.fields['insurance_type'].widget.attrs['readonly'] = True
            self.fields['query'].widget.attrs['placeholder'] = QUERY_PLACEHOLDERS.get(insurance_type, 'Enter your insurance query...')


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