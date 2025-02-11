from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import Document
import fitz

import os
import tempfile
import uuid
import pandas as pd
import re
import tempfile
import time

def format_docs(docs):
    """
    Format a list of Document objects into a single string.

    params:
        docs (list): A list of Document objects.

    returns:
        A string containing the text of all the documents joined by two newlines.
    """

    return "\n\n".join(doc.page_content for doc in docs)

def detect_bold_text(page):
    """
    Detects bold text in a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        None, prints the bold text in the PDF file.
    """
    formatted_document = ""
    
    blocks = page.get_text("dict")["blocks"]
    lines = [l for b in blocks for l in b["lines"]] # flatten the lines
    for b in blocks:  # iterate through the text blocks
        for l in b["lines"]:  # iterate through the text lines and the next lineb["lines"]:  # iterate through the text lines
            line_text = ""
            spans = l["spans"]
            total_words = sum(len(span["text"].strip().split()) for span in spans)

            #print(f"Total words: {total_words}")  # Debugging statement
            if total_words <= 3:
                for s in spans:  # iterate through the text spans
                    font_name = s["font"]
                    text = s["text"].strip()
                    is_bold = "bold" in font_name.lower()
                    is_capitalized = text.isupper()
                    num_words = len(text.split())
                    is_isolated = all(not span["text"].strip().isalpha() for span in spans if span != s)
                    is_empty = text == ""
                    if (is_bold or is_capitalized) and not is_empty:
                        #formatted_document += f"**{text}** "
                        line_text += f"**{text}** "
                        
                    else:
                        #formatted_document += text.strip() + " "
                        line_text += text + " "
            else:
                for s in spans:  # iterate through the text spans
                    text = s["text"].strip()
                    line_text += text + " "
            print(f"Line: {line_text.strip()}, number of words: {len(line_text.split())}")  # Debugging statement
            formatted_document += line_text.strip() + "\n"
    
    #print(f"Formatted document: {formatted_document.strip()}")  # Debugging statement
    #print(f"Bold text found: {bold_text_list}") # debug
    return formatted_document.strip()

    

def get_pdf_text_and_metadata(uploaded_file):
    """
    Extract text and metadata from a PDF file

    params:
        uploaded_file (streamlit.uploaded_file_manager.UploadedFile): The PDF file to extract text and metadata from

    returns:
        documents (list): A list of Document objects containing the text and metadata of the PDF file
    """
    print("Extracting text and metadata from PDF file...")
    temp_file = None
    try:
        # read file contents
        input_file = uploaded_file.read()

        if not input_file:
            raise ValueError("Uploaded file is empty")

        # create a temporary file to store the PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        if os.path.getsize(temp_file.name) == 0:
            raise ValueError("Temporary file is empty after writing")

        # load PDF document
        with fitz.open(temp_file.name) as pdf_document:
            #pdf_document = pymupdf.open(temp_file.name)
            documents = []
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                updated_document = detect_bold_text(page)
        #print(f"metadata: {metadata}") # debug
        print(f"Extracted {len(documents)} documents from PDF file") # debug
        #print(f"Extracted document: {documents}") # debug
        return documents
    
    finally:
        # delete the temporary file
        if temp_file is not None:
            try:
                time.sleep(1)  # Small delay to ensure the file handle is released
                os.unlink(temp_file.name)
            except PermissionError as e:
                print(f"PermissionError: {e}")
                print(f"Failed to delete temporary file: {temp_file.name}")

def extract_sections_and_parts(documents):
    """
    Extract sections and parts from a list of Document objects

    params:
        documents (list): A list of Document objects to extract sections and parts from

    returns:
        None, prints the extracted sections and parts in a file
    """
    print("Extracting sections and parts...")
    sections = {}
    #section_pattern = r"(?m)^(?:[A-Z][A-Z\s]+[_]+|[A-Z][A-Z\s]+$)" # regex pattern to match section headers, currently matches all caps words
    section_pattern = r"^\*\*(\w+\s?\w*\s?\w*)\*\*$"  # regex pattern to match bold text with a maximum of 3 words
    #section_pattern = r"^\*\*(.*?)\*\*$"  # regex pattern to match bold text
    text = format_docs(documents)

    #print(f"Formatted text: {text}") # debug

    matches = list(re.finditer(section_pattern, text, re.MULTILINE))  # find all matches of the pattern in the text

    print(f"Matches found: {len(matches)}")  # Debugging statement
    for match in matches:
        print(f"Match: {match.group(0)} at position {match.start()} to {match.end()}")  # Debugging statement

    for i, match in enumerate(matches):
        section_header = match.group(1)
        start_index = match.end()
        end_index = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_content = text[start_index:end_index].strip()
        sections[section_header] = section_content

        print(f"Section: {section_header}\nContent:\n{section_content}\n{'-' * 80}\n")  # Debugging statement

    output_dir = "../test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sections_and_parts.txt")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Extracted sections and parts:\n")
        for section, content in sections.items():
            f.write(f"Section: {section}\n")
            f.write(f"Content:\n{content}\n")
            f.write('-' * 80 + '\n')
    f.close()

    return sections