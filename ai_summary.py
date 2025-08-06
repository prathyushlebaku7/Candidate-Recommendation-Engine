import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_ai_summary(job_description, resume_text, candidate_name):
    prompt = f"""Job Description:{job_description}
   
    Candidate Resume Summary:{resume_text[:2000]}
    The candidate's resume file is in the pdf uploaded.
    If the filename includes words like 'resume', 'cv', or underscores, ignore them.
    Start the summary with the candidate's real name and Write a short, professional 2–3 sentence note explaining why this person is a great fit for the role.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.5
    )

    return response.choices[0].message.content.strip()
