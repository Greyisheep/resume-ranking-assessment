from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import chardet

# Load models
ranking_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v3')
summarization_model = pipeline('summarization', model='facebook/bart-large-cnn')

def chunk_text(text: str, max_tokens: int = 512):
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

def rank_cvs(job_description: str, cvs: list):
    job_embedding = ranking_model.encode(job_description)
    ranked_cvs = []

    for cv in cvs:
        if isinstance(cv, bytes):
            detected_encoding = chardet.detect(cv)['encoding']
            if detected_encoding:
                cv = cv.decode(detected_encoding, errors='ignore')
            else:
                cv = cv.decode('utf-8', errors='ignore')  # Fallback to 'utf-8' if encoding not detected

        chunks = chunk_text(cv)
        scores = []
        for chunk in chunks:
            chunk_embedding = ranking_model.encode(chunk)
            score = util.cos_sim(job_embedding, chunk_embedding)[0].item()
            scores.append(score)
        average_score = sum(scores) / len(scores)
        ranked_cvs.append({"cv_text": cv, "score": average_score})

    ranked_cvs = sorted(ranked_cvs, key=lambda x: x['score'], reverse=True)
    return ranked_cvs

def summarize_cv(cv_text: str):
    chunks = chunk_text(cv_text)
    summaries = []
    for chunk in chunks:
        summary = summarization_model(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ' '.join(summaries)
