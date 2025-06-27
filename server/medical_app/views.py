from django.shortcuts import render, redirect
from django.http import FileResponse, Http404
from django.contrib import messages
from .models import MedicalRecord
from ai_modules.summarizer.processor import summarize_and_convert

import fitz  
import docx
import os

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    elif uploaded_file.name.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

def index(request):
    if request.method == 'POST':
        patient_name = request.POST.get('patient_name', '').strip()
        uploaded_file = request.FILES.get('report_file')
        report_text = request.POST.get('report_text', '').strip()

        if uploaded_file:
            report_text = extract_text_from_file(uploaded_file)

        # Replace <n> with newline
        report_text = report_text.replace("<n>", "\n")

        if not report_text:
            return render(request, 'index.html', {
                'records': MedicalRecord.objects.all().order_by('-uploaded_at'),
                'error': 'No input found. Please upload or type a report.'
            })

        # Create initial record to generate record ID
        record = MedicalRecord.objects.create(
            patient_name=patient_name,
            report_text=report_text,
            summary_doctor="",     # updated now PMM
            summary_patient="",    # updated by PMM
            markdown_summary=""    # updated by PMM
        )

        summary_doctor, summary_patient, markdown_summary = summarize_and_convert(report_text, record.id)

        record.summary_doctor = summary_doctor
        record.summary_patient = summary_patient
        record.markdown_summary = markdown_summary
        record.save()

        return redirect('index')

    records = MedicalRecord.objects.all().order_by('-uploaded_at')
    return render(request, 'index.html', {'records': records})

def download_markdown(request, record_id):
    file_path = f"downloads/record_{record_id}.md"
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=f"record_{record_id}.md")
    else:
        raise Http404("Markdown file not found.")

def delete_all_summaries(request):
    if request.method == 'POST':
        MedicalRecord.objects.all().delete()

        download_dir = "downloads"
        if os.path.exists(download_dir):
            for filename in os.listdir(download_dir):
                if filename.endswith(".md"):
                    os.remove(os.path.join(download_dir, filename))

        messages.success(request, "All summaries deleted.")
        return redirect('index')
    else:
        raise Http404("Invalid request method.")
