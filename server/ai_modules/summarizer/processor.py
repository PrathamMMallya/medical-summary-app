from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import markdownify
import torch
import os

# Path to Pegasus model
model_path = r"C:\Users\prath\Downloads\medical_summary_project\server\medical_app\pegasus"

tokenizer = PegasusTokenizer.from_pretrained(model_path)
model = PegasusForConditionalGeneration.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_and_convert(text, record_id):
    markdown_version = markdownify.markdownify(text, heading_style="ATX")

    # Save Markdown file
    output_path = f"downloads/record_{record_id}.md"
    os.makedirs("downloads", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_version)

    inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt").to(device)
    summary_ids = model.generate(
        **inputs,
        max_length=800,       # Much longer summary
        min_length=250,
        num_beams=5,
        length_penalty=1.0,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
