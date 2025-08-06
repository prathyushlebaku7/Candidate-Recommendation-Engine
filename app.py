import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from ai_summary import generate_ai_summary

@st.cache_resource 
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

st.title("AI-Powered Resume Matching App")

job_description = st.text_area("Paste the Job Description here:", height=200)

uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDFs)", 
    type=["pdf"], 
    accept_multiple_files=True
)

if st.button("Match Resumes") and job_description and uploaded_files:
    job_embedding = model.encode([job_description])

    candidates = []
    
    progress = st.progress(0)
    
    for idx, file in enumerate(uploaded_files):
        resume_text = extract_text_from_pdf(file)  #extraction
        resume_embedding = model.encode([resume_text]) #embeddings
        similarity = cosine_similarity(job_embedding, resume_embedding)[0][0] #similarity score
        
        candidates.append({
            "Candidate (PDF Name)": file.name,
            "Resume Text": resume_text,
            "Similarity Score": round(similarity, 4)
        })
        
        progress.progress((idx + 1) / len(uploaded_files))
    
    # Sort candidates by similarity (high to low) and take top 10
    candidates_df = pd.DataFrame(candidates).sort_values(
        by="Similarity Score", ascending=False
    ).head(10)

    candidates_df.insert(0, "Rank", range(1, len(candidates_df) + 1))

    summaries = []
    with st.spinner("Generating AI summaries for top candidates..."):
        for _, row in candidates_df.iterrows():
            summary = generate_ai_summary(
                job_description, 
                row["Resume Text"], 
                row["Candidate (PDF Name)"]
            )
            summaries.append(summary)
    
    candidates_df["AI Summary"] = summaries
    candidates_df = candidates_df.drop(columns=["Resume Text"])
    candidates_df = candidates_df.reset_index(drop=True)

    # Highlighting top 5 candidates
    def highlight_top5(row):
        if row.Rank <= 5:
            return ['background-color: lightgreen' for _ in row]
        else:
            return ['background-color: lightcoral' for _ in row]

    st.subheader("Top Matching Candidates with AI Summary")
    st.dataframe(candidates_df.style.apply(highlight_top5, axis=1), use_container_width=True)
