import streamlit as st
import requests
import pandas as pd

# FastAPI endpoints
RANK_ENDPOINT = "http://localhost:8000/rank"
SUMMARIZE_ENDPOINT = "http://localhost:8000/summarize"

# Page configuration with a wide layout
st.set_page_config(layout="wide", page_title="CV Ranking and Summarization Tool")

# Custom CSS for background and general styling
st.markdown("""
    <style>
        body {
            background-color: #f0f4f7;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 20px;
            margin-top: 10px;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .st-expander {
            background-color: #f9f9f9;
        }
        .stFileUploader {
            background-color: #e6ecf0;
            padding: 10px;
            border-radius: 8px;
        }
        .stFileUploader label {
            font-weight: bold;
            color: #4B0082;
        }
        .st-cp .css-15tx938 p {
            font-size: 1.2em;
        }
        .stMarkdown h1 {
            font-size: 2.5em;
            font-weight: bold;
            color: #4B0082;
            text-align: center;
            text-shadow: 2px 2px 5px #aaa;
        }
        hr {
            border: 1px solid #4B0082;
        }
        .dataframe {
            width: 100% !important;
            margin: 20px auto;
        }
        .dataframe th, .dataframe td {
            text-align: center !important;
            padding: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Header and description
html_code = """
<div style="text-align: center; font-size: 2.5em; font-weight: bold; margin-top: 20px; margin-bottom: 20px; color: #4B0082; text-shadow: 2px 2px 5px #aaa;">
    CV Ranking and Summarization Tool
</div>
"""

st.markdown(html_code, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; font-size: 1.2em; margin-bottom: 30px;">
    Upload a job description and multiple CVs to rank them based on the job requirements. 
    Additionally, you can generate summaries of each CV to quickly grasp key qualifications.
</div>
""", unsafe_allow_html=True)

# File Upload Section
col1, col2, col3 = st.columns([1, 0.1, 1])

with col1:
    job_description = st.file_uploader("Upload Job Description", type=["pdf"])

with col3:
    cvs = st.file_uploader("Upload CVs", type=["pdf"], accept_multiple_files=True)

# Button Section with Icons
col1, col2, col3 = st.columns([1, 0.1, 1])

with col1:
    rank_btn = st.button("🔍 Rank CVs")
with col3:
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
            df.insert(0, 'Rank', range(1, len(df) + 1))  # Add a Rank column starting from 1
            with st.expander("View Ranked CVs"):
                st.dataframe(df[['Rank', 'filename', 'score']], use_container_width=True)
        else:
            st.error("Error ranking CVs. Please check your inputs and try again.")
    else:
        st.warning("Please upload both a job description and at least one CV.")

if summarize_btn:
    if job_description and cvs:
        for cv in cvs:
            files = {
                "job_description": (job_description.name, job_description.getvalue(), "application/pdf"),
                "file": (cv.name, cv.getvalue(), "application/pdf")
            }
            response = requests.post(SUMMARIZE_ENDPOINT, files=files)
            if response.status_code == 200:
                summary = response.json()["summary"]
                with st.expander(f"Summary for {cv.name}"):
                    st.write(summary)
            else:
                st.error(f"Error summarizing {cv.name}. Please try again.")
    else:
        st.warning("Please upload both a job description and at least one CV.")

# Footer with Contact Information
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-top: 20px;">
    For more tools and updates, visit <a href="https://github.com/greyisheep" style="color: #4B0082;">GitHub</a> or contact me at <a href="mailto:ibeawuchiclaret@gmail.com" style="color: #4B0082;">ibeawuchiclaret@gmail.com</a>.
</div>
""", unsafe_allow_html=True)
