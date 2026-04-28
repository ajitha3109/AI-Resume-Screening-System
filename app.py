import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------- FUNCTIONS -------- #

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    return re.sub(r'\W', ' ', text.lower())

def match_score(resume, jd):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, jd])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

def extract_skills(text, skills):
    return [skill for skill in skills if skill in text]

def skill_score(found, required):
    return len(found)/len(required)

def extract_name(text):
    lines = text.strip().split("\n")
    return lines[0] if lines else "Unknown Candidate"

# -------- UI -------- #

st.set_page_config(page_title="AI Resume Screening System", layout="centered")

st.title("AI Resume Screening System")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)", 
    type="pdf", 
    accept_multiple_files=True
)

job_desc = st.text_area("Enter Job Description")

skills = ["python", "machine learning", "nlp", "sql", "data analysis"]

if uploaded_files and job_desc:

    jd = clean_text(job_desc)
    results = []

    for file in uploaded_files:
        raw_text = extract_text(file)
        name = extract_name(raw_text)
        resume = clean_text(raw_text)

        score = match_score(resume, jd)
        found = extract_skills(resume, skills)
        final = (score * 0.7) + (skill_score(found, skills) * 0.3)

        results.append((name, final, found))

    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)

    st.subheader("🏆 Candidate Ranking")

    for name, score, found in results:
        st.markdown(f"### 👤 {name}")
        st.write(f"📊 Score: {round(score*100,2)}%")

        st.write("✅ Skills Found:")
        for s in found:
            st.write(f"✔️ {s}")

        missing = list(set(skills) - set(found))
        st.write("❌ Missing Skills:")
        for m in missing:
            st.write(f"❌ {m}")

        if score > 0.5:
            st.success("Selected")
        else:
            st.error("Not Selected")

        st.write("---")