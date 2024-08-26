from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from app.models import rank_cvs, summarize_cv
from app.utils import convert_pdf_to_text
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/rank")
async def rank_cvs_endpoint(
    job_description: UploadFile = File(...), 
    files: List[UploadFile] = File(...)
):
    try:
        logger.info("Received request to rank CVs")

        # Log file details for debugging (optional)
        logger.debug(f"Job description filename: {job_description.filename}")
        logger.debug(f"CV filenames: {[file.filename for file in files]}")

        job_description_text = convert_pdf_to_text([job_description.file])[0]
        cv_texts = [convert_pdf_to_text([file.file])[0] for file in files]
        filenames = [file.filename for file in files]
        ranked_cvs = rank_cvs(job_description_text, cv_texts, filenames)
        
        logger.info("CV ranking completed successfully")
        return {"ranked_cvs": ranked_cvs}

    except Exception as e:
        logger.error(f"Error in ranking: {e}") 
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="An error occurred while processing the files. Please check the file formats and try again."
        )

@app.post("/summarize")
async def summarize_cv_endpoint(file: UploadFile = File(...)):
    try:
        logger.info(f"Received request to summarize CV: {file.filename}")
        cv_text = convert_pdf_to_text([file.file])[0]
        summary = summarize_cv(cv_text)

        logger.info("CV summarization completed successfully")
        return {"summary": summary}

    except Exception as e:
        logger.error(f"Error in summarizing: {e}") 
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, 
            detail=f"An error occurred while summarizing {file.filename}. Please check the file format and try again."
        )
