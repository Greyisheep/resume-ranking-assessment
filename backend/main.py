from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from typing import List
import logging

from backend.models import rank_cvs, summarize_cv
from backend.utils import convert_pdf_to_text
from backend.metrics import evaluate_and_log_mrr, evaluate_and_log_ndcg, log_bert_scores

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/rank")
async def rank_cvs_endpoint(
    job_description: UploadFile = File(...), 
    files: List[UploadFile] = File(...)
):
    """
    Endpoint to rank CVs based on a job description.

    Args:
        job_description (UploadFile): The job description file uploaded by the user.
        files (List[UploadFile]): A list of CV files uploaded by the user.

    Returns:
        dict: A dictionary containing the ranked CVs.

    Raises:
        HTTPException: If an error occurs while processing the files.
    """
    try:
        logger.info("Received request to rank CVs")
        job_description_text = convert_pdf_to_text([job_description.file])[0]
        cv_texts = [convert_pdf_to_text([file.file])[0] for file in files]
        filenames = [file.filename for file in files]

        ranked_cvs = rank_cvs(job_description_text, cv_texts, filenames)

        # Log both MRR and NDCG scores using weighted scoring with a dynamic threshold
        evaluate_and_log_mrr(ranked_cvs, logger)
        evaluate_and_log_ndcg(ranked_cvs, logger)
        
        logger.info("CV ranking completed successfully")
        return {"ranked_cvs": ranked_cvs}

    except Exception as e:
        logger.error(f"Error in ranking CVs: {e}")
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="An error occurred while processing the files. Please check the file formats and try again."
        )
@app.post("/summarize")
async def summarize_cvs_endpoint(
    job_description: UploadFile = File(...), 
    files: List[UploadFile] = File(...)
):
    """
    Endpoint to summarize multiple CVs based on a job description.

    Args:
        job_description (UploadFile): The job description file uploaded by the user.
        files (List[UploadFile]): A list of CV files uploaded by the user.

    Returns:
        dict: A dictionary containing the summaries of the CVs.

    Raises:
        HTTPException: If an error occurs during the summarization process.
    """
    try:
        logger.info("Received request to summarize multiple CVs")
        job_description_text = convert_pdf_to_text([job_description.file])[0]
        summaries = {}
        
        for file in files:
            cv_text = convert_pdf_to_text([file.file])[0]
            summary = summarize_cv(cv_text, job_description_text)
            summaries[file.filename] = summary

            # Log BERTScore for each summarization
            log_bert_scores(summary, logger)
        
        logger.info("CV summarization completed successfully")
        return {"summaries": summaries}

    except Exception as e:
        logger.error(f"Error in summarizing CVs: {e}")
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="An error occurred while summarizing the CVs. Please check the file formats and try again."
        )