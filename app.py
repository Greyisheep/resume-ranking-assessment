import streamlit as st
import requests
import pandas as pd

# FastAPI endpoints
RANK_ENDPOINT = "http://localhost:8000/rank"
SUMMARIZE_ENDPOINT = "http://localhost:8000/summarize"

# Header and description
st.title("CV Ranking and Summarization Tool")
st.markdown("""
Upload a job description and multiple CVs to rank them based on the job requirements. 
Additionally, you can generate summaries of each CV to quickly grasp key qualifications.
""")

# File Upload Section
col1, col2 = st.columns(2)

with col1:
    job_description = st.file_uploader("Upload Job Description", type=["pdf"])

with col2:
    cvs = st.file_uploader("Upload CVs", type=["pdf"], accept_multiple_files=True)

# Button Section with Icons
col1, col2 = st.columns(2)

with col1:
    rank_btn = st.button("🔍 Rank CVs")
with col2:
    summarize_btn = st.button("📝 Summarize CVs")

# Results Section
if rank_btn:
    if job_description and cvs:
        # Prepare files for the request
        files = [
            ("job_description", (job_description.name, job_description.getvalue(), "application/pdf"))
        ]
        files.extend(
            ("files", (cv.name, cv.getvalue(), "application/pdf")) for cv in cvs
        )

        # Send request to rank endpoint
        response = requests.post(RANK_ENDPOINT, files=files)
        
        if response.status_code == 200:
            ranked_cvs = response.json()["ranked_cvs"]
            df = pd.DataFrame(ranked_cvs)
            df['score'] = df['score'].map("{:.2f}%".format)  # Format scores as percentages
            with st.expander("View Ranked CVs"):
                st.dataframe(df[['filename', 'score']])
        else:
            st.error("Error ranking CVs. Please check your inputs and try again.")
    else:
        st.warning("Please upload both a job description and at least one CV.")

if summarize_btn:
    if cvs:
        for cv in cvs:
            files = {"file": (cv.name, cv.getvalue(), "application/pdf")}
            response = requests.post(SUMMARIZE_ENDPOINT, files=files)
            if response.status_code == 200:
                summary = response.json()["summary"]
                with st.expander(f"Summary for {cv.name}"):
                    st.write(summary)
            else:
                st.error(f"Error summarizing {cv.name}. Please try again.")
    else:
        st.warning("Please upload at least one CV.")

# Footer with Contact Information
st.markdown("---")
st.markdown("For more tools and updates, visit [GitHub](https://github.com/greyisheep) or contact me at ibeawuchiclaret@gmail.com.")

# Custom CSS Styling
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    font-size: 16px;
}
.st-expander {
    background-color: #f9f9f9;
}
</style>
""", unsafe_allow_html=True)
