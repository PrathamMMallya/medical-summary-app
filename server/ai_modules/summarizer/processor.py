from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import markdownify
import torch
import os

# Path to local Pegasus model
model_path = r"C:\Users\tjsre\Desktop\projects\practice\ml\navy_project\secret\models\pegasus"

# Load tokenizer and model once
tokenizer = PegasusTokenizer.from_pretrained(model_path)
model = PegasusForConditionalGeneration.from_pretrained(model_path).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
model.eval()


# Summary generator function
def generate_summary(text, prompt):
    device = model.device
    input_text = f"{prompt.strip()}\n\n{text.strip()}"

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


# Main processor
def summarize_and_convert(text, record_id):
    cleaned_text = text.replace("<n>", "\n").strip()

    # Generate a unified insurance summary
    insurance_prompt = (
        "Summarize the following document to extract key insurance-related information. "
        "If it's a health report, include age, medical conditions, treatments, and budget. "
        "If it's about a vehicle, extract vehicle type, accident history, repairs, coverage preferences, and cost details."
    )
    summary_insurance = generate_summary(cleaned_text, insurance_prompt)

    # Remove any boilerplate pattern if accidentally retained
    for unwanted_prefix in [
        "Key details for health insurance", 
        "summary should help an insurance recommender", 
        "Summarize the following document"
    ]:
        if unwanted_prefix.lower() in summary_insurance.lower():
            parts = summary_insurance.split("\n")
            filtered = [line for line in parts if unwanted_prefix.lower() not in line.lower()]
            summary_insurance = "\n".join(filtered).strip()

    # Generate Markdown version of the original report
    markdown_version = markdownify.markdownify(cleaned_text, heading_style="ATX")

    os.makedirs("downloads", exist_ok=True)
    with open(f"downloads/record_{record_id}.md", "w", encoding="utf-8") as f:
        f.write(markdown_version)

    return summary_insurance, markdown_version
