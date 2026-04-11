"""Read the pre-defense PDF and extract text."""
import fitz  # PyMuPDF

pdf_path = "/mnt/c/Users/USERAS/Downloads/pptx pre defense.pdf"
doc = fitz.open(pdf_path)
print(f"Total pages: {doc.page_count}\n")

for i in range(doc.page_count):
    page = doc[i]
    text = page.get_text()
    if text.strip():
        print(f"\n{'='*60}")
        print(f"PAGE {i+1}")
        print(f"{'='*60}")
        print(text[:2000])
        if len(text) > 2000:
            print(f"\n... [{len(text)-2000} more characters truncated]")
