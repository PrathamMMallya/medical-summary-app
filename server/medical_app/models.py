# medical_app/models.py
from django.db import models

class MedicalRecord(models.Model):
    patient_name = models.CharField(max_length=100)
    report_text = models.TextField()
    insurance_summary = models.TextField()
    markdown_summary = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    insurance_type = models.CharField(max_length=50, choices=[
        ('health', 'Health Insurance'),
        ('car', 'Car Insurance'),
        ('life', 'Life Insurance'),
        ('home', 'Home Insurance'),
    ], default='health')

    def __str__(self):
        return f"{self.patient_name} ({self.insurance_type})"