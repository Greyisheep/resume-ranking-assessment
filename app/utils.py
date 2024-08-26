from pdfminer.high_level import extract_text
from typing import List

def convert_pdf_to_text(pdf_paths: List[str]) -> List[str]:
    """
    Convert a list of PDF file paths to a list of extracted text strings.

    Args:
        pdf_paths (List[str]): A list of paths to PDF files.

    Returns:
        List[str]: A list of strings where each string contains the text 
                   extracted from the corresponding PDF file.
    """
    texts = []
    for pdf_path in pdf_paths:
        try:
            text = extract_text(pdf_path)
            texts.append(text)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            texts.append("")
    return texts
