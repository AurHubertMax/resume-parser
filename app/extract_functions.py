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
import pymupdf

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

def get_pdf_text_and_metadata(uploaded_file):
    """
    Extract text and metadata from a PDF file

    params:
        uploaded_file (streamlit.uploaded_file_manager.UploadedFile): The PDF file to extract text and metadata from

    returns:
        documents (list): A list of Document objects containing the text and metadata of the PDF file
    """
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
        pdf_document = pymupdf.open(temp_file.name)
        documents = []

        # extract text and metadata from each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text_dict = page.get_text("dict")
            text = ""
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["flags"] & 2: # check if text is bold
                                # check if the line has a max of 3 words and no other text after it
                                words = span["text"].strip().split()
                                if len(words) <= 3 and re.match(r"^\*\*(\w+\s?\w*\s?\w*)\*\*$", span["text"].strip()):
                                    text += f"**{span['text'].strip()}**\n"
                                else:
                                    text += span["text"]
                            else:
                                text += span["text"]
                        text += "\n"
            
            metadata = pdf_document.metadata
            documents.append(Document(page_content=text, metadata=metadata))
        
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
    sections = {}
    #section_pattern = r"(?m)^(?:[A-Z][A-Z\s]+[_]+|[A-Z][A-Z\s]+$)" # regex pattern to match section headers, currently matches all caps words
    section_pattern = r"^\*\*(\w+\s?\w*\s?\w*)\*\*$"  # regex pattern to match bold text with a maximum of 3 words
    text = format_docs(documents)
    matches = list(re.finditer(section_pattern, text, re.MULTILINE))  # find all matches of the pattern in the text

    for i, match in enumerate(matches):
        section_header = match.group(1)
        start_index = match.end()
        end_index = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_content = text[start_index:end_index].strip()
        sections[section_header] = section_content

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

    # return sections