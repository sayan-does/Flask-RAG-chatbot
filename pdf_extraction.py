import fitz  # in the module PyMuPDF


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


# Specify your PDF file path
pdf_path = r"I:\interview_proj\comp2-altruism\SC_data.pdf"
# Specify the text file path where you want to save the extracted text
output_text_file_path = "output_text_file.txt"

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Save the extracted text to a file
save_text_to_file(pdf_text, output_text_file_path)

print(f"Text extracted and saved to {output_text_file_path}")
