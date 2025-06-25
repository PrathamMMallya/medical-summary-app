from django.db import models

class MedicalRecord(models.Model):
    patient_name = models.CharField(max_length=100)
    report_text = models.TextField()
    summary = models.TextField(blank=True, null=True)
    markdown_summary = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.patient_name
