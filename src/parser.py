import PyPDF2
from docx import Document  # ✅ Correct import
from io import BytesIO

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDF/DOCX/TXT files"""
    try:
        if filename.lower().endswith('.pdf'):
            reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        
        elif filename.lower().endswith('.docx'):
            doc = Document(BytesIO(file_bytes))  # ✅ Works with python-docx
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        
        elif filename.lower().endswith('.txt'):
            return file_bytes.decode('utf-8', errors='ignore')
        
        raise ValueError(f"Unsupported file type: {filename}")
    
    except Exception as e:
        raise ValueError(f"Text extraction failed: {str(e)}")

