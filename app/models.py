from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import chardet

# Load models (using specific model names for better control)
ranking_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v3')
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
summarization_model = pipeline('summarization', model=model, tokenizer=tokenizer)

def chunk_text(text: str, max_tokens: int = 512) -> list:
    """
    Splits text into chunks of a specified maximum token length.

    Args:
        text (str): The text to be split into chunks.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        list: A list of text chunks.
    """
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append('. '.join(current_chunk))

    return chunks

def rank_cvs(job_description: str, cvs: list, filenames: list) -> list:
    """
    Ranks CVs based on their relevance to the job description.

    Args:
        job_description (str): The job description text.
        cvs (list): List of CV texts.
        filenames (list): List of filenames corresponding to the CVs.

    Returns:
        list: A list of dictionaries containing filenames and their relevance scores.
    """
    job_embedding = ranking_model.encode(job_description)
    ranked_cvs = []

    for cv, filename in zip(cvs, filenames):
        if isinstance(cv, bytes):
            detected_encoding = chardet.detect(cv)['encoding']
            if detected_encoding:
                cv = cv.decode(detected_encoding, errors='ignore')
            else:
                cv = cv.decode('utf-8', errors='ignore')  # Fallback to 'utf-8' if encoding not detected

        cv_embedding = ranking_model.encode(cv)
        score = util.pytorch_cos_sim(job_embedding, cv_embedding).item()
        percentage_score = score * 100  # Convert to percentage
        ranked_cvs.append({'filename': filename, 'score': percentage_score})

    ranked_cvs = sorted(ranked_cvs, key=lambda x: x['score'], reverse=True)
    return ranked_cvs

def summarize_cv(cv_text: str) -> str:
    """
    Summarizes the given CV text.

    Args:
        cv_text (str): The CV text to summarize.

    Returns:
        str: The summarized text of the CV.
    """
    chunks = chunk_text(cv_text)
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)
        summary_ids = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            max_length=130, 
            min_length=30, 
            do_sample=False
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return ' '.join(summaries)
