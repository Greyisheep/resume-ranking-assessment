from pdfminer.high_level import extract_text
from typing import List

def convert_pdf_to_text(pdf_paths: List[str]) -> List[str]:
    texts = []
    for pdf_path in pdf_paths:
        text = extract_text(pdf_path)
        texts.append(text)
    return texts
