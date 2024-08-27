from concurrent.futures import ThreadPoolExecutor

from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import chardet
from nltk.corpus import stopwords
import nltk
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load models
ranking_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v3')
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
summarization_model = pipeline('summarization', model=model, tokenizer=tokenizer)

def remove_stopwords(text: str) -> str:
    return ' '.join(word for word in text.split() if word.lower() not in stop_words)

def clean_text(text: str) -> str:
    """
    Cleans the input text by performing common OCR corrections.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # Common OCR corrections
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = text.replace('-\n', '')  # Fix hyphenated line breaks
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()

def chunk_text(text: str, max_tokens: int = 1024) -> list:
    """
    Splits the input text into chunks, each containing up to a specified maximum number of tokens.

    Args:
        text (str): The input text to be chunked.
        max_tokens (int, optional): The maximum number of tokens per chunk. Defaults to 1024.

    Returns:
        list: A list of text chunks, each containing up to max_tokens tokens.
    """
    tokens = text.split()
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def generate_summary(text: str) -> str:
    """
    Generates a summary for the given text using a pre-trained model.

    Args:
        text (str): The input text to be summarized.

    Returns:
        str: The generated summary of the input text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)
    summary_ids = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        max_length=100,
        min_length=50,
        do_sample=False
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # Remove any extra spaces between words
    summary = re.sub(r'\s+', ' ', summary)
    return summary.strip()

def rank_cvs(job_description: str, cvs: list, filenames: list) -> list:
    """
    Ranks a list of CVs based on their similarity to a given job description.

    Args:
        job_description (str): The job description to compare the CVs against.
        cvs (list): A list of CVs, where each CV is either a string or bytes.
        filenames (list): A list of filenames corresponding to the CVs.

    Returns:
        list: A list of dictionaries, each containing the filename and the similarity score,
              sorted by the score in descending order.
    """
    job_description = remove_stopwords(job_description)
    job_embedding = ranking_model.encode(job_description)
    ranked_cvs = []

    def process_cv(cv, filename):
        """
        Processes a single CV by removing stopwords, encoding it, and calculating its similarity score.

        Args:
            cv (str or bytes): The CV to process.
            filename (str): The filename of the CV.

        Returns:
            dict: A dictionary containing the filename and the similarity score.
        """
        if isinstance(cv, bytes):
            detected_encoding = chardet.detect(cv)['encoding']
            if detected_encoding:
                cv = cv.decode(detected_encoding, errors='ignore')
            else:
                cv = cv.decode('utf-8', errors='ignore')

        cv = remove_stopwords(cv)
        cv_embedding = ranking_model.encode(cv)
        score = util.pytorch_cos_sim(job_embedding, cv_embedding).item()
        percentage_score = score * 100  # Convert to percentage
        return {'filename': filename, 'score': percentage_score}

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_cv, cvs, filenames)
    
    ranked_cvs = sorted(results, key=lambda x: x['score'], reverse=True)
    return ranked_cvs

def summarize_chunk(chunk: str, job_description: str) -> tuple:
    """
    Summarizes a text chunk by extracting relevant skills and experiences based on a job description.

    Args:
        chunk (str): The text chunk to be summarized.
        job_description (str): The job description used to identify relevant skills and experiences.

    Returns:
        tuple: A tuple containing two summaries:
            - skills_summary (str): A summary of relevant skills.
            - experience_summary (str): A summary of relevant experiences.
    """
    sentences = chunk.split('. ')
    relevant_skills = []
    relevant_experiences = []

    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in job_description.lower().split()):
            if "experience" in sentence.lower() or "worked" in sentence.lower():
                relevant_experiences.append(sentence)
            else:
                relevant_skills.append(sentence)

    skills_summary = generate_summary(' '.join(relevant_skills)) if relevant_skills else ""
    experience_summary = generate_summary(' '.join(relevant_experiences)) if relevant_experiences else ""

    return (skills_summary, experience_summary)

def summarize_cv(cv_text: str, job_description: str) -> str:
    """
    Summarizes the given CV text based on the provided job description.

    This function cleans the CV text, breaks it into chunks, and then uses a thread pool
    to concurrently summarize each chunk. The summaries of skills and experiences are
    then combined into a final summary.

    Args:
        cv_text (str): The text of the CV to be summarized.
        job_description (str): The job description to tailor the summary towards.

    Returns:
        str: A summarized version of the CV, focusing on relevant skills and experiences.
    """
    cv_text = clean_text(cv_text)
    chunks = chunk_text(cv_text)
    skills_summary = []
    experience_summary = []

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda chunk: summarize_chunk(chunk, job_description), chunks)

    for skills, experiences in results:
        if skills:
            skills_summary.append(skills)
        if experiences:
            experience_summary.append(experiences)

    skills_paragraph = ' '.join(skills_summary)
    experience_paragraph = ' '.join(experience_summary)

    return f"{skills_paragraph}\n\n{experience_paragraph}"