from django.db import models

class MedicalRecord(models.Model):
    patient_name = models.CharField(max_length=100)
    report_text = models.TextField()
    insurance_summary = models.TextField()   # Unified summary used for insurance recommendation
    markdown_summary = models.TextField()    # Original input converted to markdown
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.patient_name
