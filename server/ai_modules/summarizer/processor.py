from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import markdownify
import torch
import os

# Path to Pegasus model
model_path = r"C:\Users\prath\Downloads\medical_summary_project\server\medical_app\pegasus"

# Load Pegasus model
tokenizer = PegasusTokenizer.from_pretrained(model_path)
model = PegasusForConditionalGeneration.from_pretrained(model_path).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
model.eval()

# Detect insurance type from text
def detect_insurance_type(text: str) -> str:
    text_lower = text.lower()
    if any(word in text_lower for word in ['hospital', 'treatment', 'diagnosis', 'symptom', 'fever', 'medicine', 'bp', 'diabetes']):
        return 'health'
    elif any(word in text_lower for word in ['vehicle', 'car', 'accident', 'rc', 'license', 'repair', 'service', 'claim']):
        return 'vehicle'
    elif any(word in text_lower for word in ['life insurance', 'term plan', 'lic', 'family protection', 'corpus', 'retirement', 'bmi', 'smoker']):
        return 'life'
    return 'unknown'

# Generate summary from just the record text
def generate_summary(text, prompt):
    device = model.device
    input_text = f"{text.strip()}"  # Only pass record content, NOT prompt

    inputs = tokenizer(
        input_text,
        truncation=True,
        padding="longest",
        max_length=1024,
        return_tensors="pt"
    ).to(device)

    summary_ids = model.generate(
        **inputs,
        max_length=350,
        min_length=100,
        num_beams=4,
        length_penalty=1.2,
        early_stopping=True,
        do_sample=False
    )

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output.strip()

# Remove unwanted instruction text from model output
def clean_summary(summary: str) -> str:
    prompt_keywords = [
        "generate a structured summary", "based on the detected insurance type", "key details",
        "this summary", "summarize", "given the following", "recommend", "include", "medical history",
        "coverage preferences", "objectives", "for health", "for life", "for vehicle", "summary should"
    ]
    lines = summary.splitlines()
    return "\n".join([
        line.strip() for line in lines
        if not any(kw in line.lower() for kw in prompt_keywords)
    ]).strip()

# Main processing function
def summarize_and_convert(text, record_id):
    cleaned_text = text.replace("<n>", "\n").strip()
    insurance_type = detect_insurance_type(cleaned_text)

    if insurance_type == "health":
        prompt = "Summarize the patient's age, medical conditions, treatments, and any budget or financial limits."
    elif insurance_type == "vehicle":
        prompt = "Summarize the vehicle type, usage, any accidents or service history, and coverage preferences."
    elif insurance_type == "life":
        prompt = "Summarize age, family medical history, risk factors, and desired life coverage or protection goals."
    else:
        prompt = "Summarize the given insurance-related information in a clear and structured way."

    # Generate summary
    summary_insurance = generate_summary(cleaned_text, prompt)
    summary_insurance = clean_summary(summary_insurance)

    # Convert full input to markdown
    markdown_version = markdownify.markdownify(cleaned_text, heading_style="ATX")

    # Save markdown file
    os.makedirs("downloads", exist_ok=True)
    with open(f"downloads/record_{record_id}.md", "w", encoding="utf-8") as f:
        f.write(markdown_version)

    return summary_insurance.replace("<n>", "\n").strip(), markdown_version.replace("<n>", "\n").strip()

