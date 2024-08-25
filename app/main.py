from fastapi import FastAPI, UploadFile, File
from app.models import rank_cvs, summarize_cv
from app.utils import convert_pdf_to_text
from typing import List

app = FastAPI()

@app.post("/rank")
async def rank_cvs_endpoint(job_description: UploadFile = File(...), files: List[UploadFile] = File(...)):
    job_description_text = convert_pdf_to_text([job_description.file])[0]
    cv_texts = [convert_pdf_to_text([file.file])[0] for file in files]
    ranked_cvs = rank_cvs(job_description_text, cv_texts)
    return {"ranked_cvs": ranked_cvs}

@app.post("/summarize")
async def summarize_cv_endpoint(file: UploadFile = File(...)):
    cv_text = convert_pdf_to_text([file.file])[0]
    summary = summarize_cv(cv_text)
    return {"summary": summary}
