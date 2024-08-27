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
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()

def chunk_text(text: str, max_tokens: int = 1024) -> list:
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
    job_description = remove_stopwords(job_description)
    job_embedding = ranking_model.encode(job_description)
    ranked_cvs = []

    for cv, filename in zip(cvs, filenames):
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
        ranked_cvs.append({'filename': filename, 'score': percentage_score})

    ranked_cvs = sorted(ranked_cvs, key=lambda x: x['score'], reverse=True)
    return ranked_cvs

def summarize_cv(cv_text: str, job_description: str) -> str:
    cv_text = clean_text(cv_text)
    chunks = chunk_text(cv_text)
    skills_summary = []
    experience_summary = []

    for chunk in chunks:
        if len(chunk.split()) < 10:  # Skip very short chunks
            continue

        sentences = chunk.split('. ')
        relevant_skills = []
        relevant_experiences = []

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in job_description.lower().split()):
                if "experience" in sentence.lower() or "worked" in sentence.lower():
                    relevant_experiences.append(sentence)
                else:
                    relevant_skills.append(sentence)

        if relevant_skills:
            summary = generate_summary(' '.join(relevant_skills))
            skills_summary.append(summary)

        if relevant_experiences:
            summary = generate_summary(' '.join(relevant_experiences))
            experience_summary.append(summary)

    skills_paragraph = ' '.join(skills_summary)
    experience_paragraph = ' '.join(experience_summary)

    return f"{skills_paragraph}\n\n{experience_paragraph}"
