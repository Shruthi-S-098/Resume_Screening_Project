from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import gradio as gr
import pandas as pd
import os

# Load MiniLM model (fully free + efficient)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF using fitz (PyMuPDF)
def extract_text_from_pdf(path):
    try:
        with fitz.open(path) as doc:
            return "
".join([page.get_text() for page in doc])
    except Exception as e:
        return f"❌ Error reading PDF: {str(e)}"

# Core matching logic
def match_resumes(jd_text, jd_pdf, resume_paths):
    try:
        # If user uploaded JD PDF, extract text
        if jd_pdf is not None:
            jd_text = extract_text_from_pdf(jd_pdf)

        if not jd_text or not resume_paths:
            return "❌ Please provide both Job Description and Resumes.", None

        # Encode job description
        jd_embedding = model.encode(jd_text, convert_to_tensor=True)

        results = []

        for path in resume_paths:
            name = os.path.basename(path)
            resume_text = extract_text_from_pdf(path)

            if resume_text.startswith("❌"):
                results.append({"Candidate": name, "Score": 0.0})
                continue

            resume_embedding = model.encode(resume_text, convert_to_tensor=True)
            score = util.cos_sim(jd_embedding, resume_embedding)[0][0].item()
            results.append({"Candidate": name, "Score": round(score, 4)})

        # Sort results by score
        df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
        best = f"🎯 Best Match: {df.iloc[0]['Candidate']} with score {df.iloc[0]['Score']}"
        return best, df

    except Exception as e:
        return f"❌ ERROR: {str(e)}", None

# Launch Gradio app
demo = gr.Interface(
    fn=match_resumes,
    inputs=[
        gr.Textbox(label="Job Description (or leave blank if uploading PDF)", lines=6),
        gr.File(label="Upload JD PDF (optional)", type="filepath", file_types=[".pdf"]),
        gr.File(label="Upload Resume PDFs (multiple)", type="filepath", file_types=[".pdf"], file_count="multiple"),
    ],
    outputs=[
        gr.Textbox(label="Top Match"),
        gr.Dataframe(label="All Matches"),
    ],
    title="📝AI CVScreen💬",
    description="Paste or upload a job description and upload multiple resumes. The AI will rank them and select the best match."
)

demo.launch()