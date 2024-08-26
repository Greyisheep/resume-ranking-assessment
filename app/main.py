from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from typing import List
import logging

from app.models import rank_cvs, summarize_cv
from app.utils import convert_pdf_to_text
from app.metrics import evaluate_and_log_mrr  # Import the evaluation function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/rank")
async def rank_cvs_endpoint(
    job_description: UploadFile = File(...), 
    files: List[UploadFile] = File(...)
):
    """
    Rank CVs based on their relevance to the job description.

    Args:
        job_description (UploadFile): The uploaded job description in PDF format.
        files (List[UploadFile]): A list of uploaded CVs in PDF format.

    Returns:
        dict: A dictionary containing the filenames and their corresponding relevance scores.
    """
    try:
        logger.info("Received request to rank CVs")
        job_description_text = convert_pdf_to_text([job_description.file])[0]
        cv_texts = [convert_pdf_to_text([file.file])[0] for file in files]
        filenames = [file.filename for file in files]

        ranked_cvs = rank_cvs(job_description_text, cv_texts, filenames)
        
        # Assume the ground truth is provided or predetermined
        # For simplicity, let's assume ground_truth is filenames sorted alphabetically
        ground_truth = sorted(filenames)

        # Evaluate and log the MRR score
        evaluate_and_log_mrr(ranked_cvs, ground_truth, logger)
        
        logger.info("CV ranking completed successfully")
        return {"ranked_cvs": ranked_cvs}

    except Exception as e:
        logger.error(f"Error in ranking CVs: {e}")
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="An error occurred while processing the files. Please check the file formats and try again."
        )


@app.post("/summarize")
async def summarize_cv_endpoint(
    job_description: UploadFile = File(...), 
    file: UploadFile = File(...)
):
    """
    Summarizes a CV into two paragraphs, highlighting key skills and experiences relevant to the job description.

    Args:
        job_description (UploadFile): The job description PDF file.
        file (UploadFile): The CV PDF file.

    Returns:
        dict: A dictionary containing the summary of the CV.
    """
    try:
        logger.info(f"Received request to summarize CV: {file.filename}")

        # Convert files to text
        job_description_text = convert_pdf_to_text([job_description.file])[0]
        cv_text = convert_pdf_to_text([file.file])[0]

        # Summarize the CV
        summary = summarize_cv(cv_text, job_description_text)

        logger.info("CV summarization completed successfully")
        return {"summary": summary}

    except Exception as e:
        logger.error(f"Error in summarizing: {e}") 
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, 
            detail=f"An error occurred while summarizing {file.filename}. Please check the file format and try again."
        )