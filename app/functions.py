from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

import os
import tempfile
import uuid
import pandas as pd
import re

# set pandas display options
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def get_pdf_text(uploaded_file):
    """
    Load a PDF document from an uploaded file and return it as a list of documents

    Params:
        uploaded_file (file-like object): The uploaded PDF file to load

    Returns:
        documents (list): A list of documents created from the uploaded PDF file
    """

    try:
        # read file contents
        input_file = uploaded_file.read()

        # create a temp file (PyPDFLoader needs a file path to read the PDF,
        # it cant work directly with a file-like object or byte streams that we get from Streamlit's uploaded_file.read())
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        # load PDF document
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()

        return documents
    
    finally:
        # Ensure the temp file is deleted after use
        os.unlink(temp_file.name)

def create_vector_store_from_texts(documents, api_key, file_name):
    """
    Create a vector store from a list of texts

    params:
        documents (list): A list of generic texts to create a vector store from
        api_key (str): The OpenAI API key used to generate the vector store
        file_name (str): The name of the file that the texts were extracted from
    
    returns:
        A Chroma vector store object
    """
    
    # step 2: split the documents into smaller chunks
    docs = split_document(documents, chunk_size=800, chunk_overlap=200)

    # step 3: define embedding function, embedding function is a function that takes a text and returns a vector
    embedding_fn = get_embedding_function(api_key)

    # step 4: create a vector store
    vector_store = create_vector_store(docs, embedding_fn, file_name)

    return vector_store

def split_document(documents, chunk_size, chunk_overlap):
    """ 
    params:
        documents (list): A list of generic texts to split into smaller chunks
        chunk_size (int): The desired max size of each chunk (default: 400)
        chunk_overlap (int): The number of characters to overlap between each chunk (default: 20)

    returns:
        list (list): A list of smaller text chunks created from the generic texts
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap, 
                                                   length_function=len, 
                                                   separators=["\n\n", "\n", " "])
    
    test_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4o-mini",
                                                                         chunk_size=chunk_size,
                                                                        chunk_overlap=chunk_overlap,
                                                                        )
    test = test_splitter.split_text(documents)
    print(test)
    return text_splitter.split_documents(documents)

def get_embedding_function(api_key):
    """
    Return an OpenAIEmbeddings object, which is used to create vector embeddings from text.
    The embeddings model used is "text-embedding-ada-002" and the OpenAI API key is provided
    as an argument to the function.

    params:
        api_key (str): The OpenAI API key to use when calling the OpenAi Embeddings API.
    
    returns:
        embeddings (OpenAiEmbeddings object): An OpenAIEmbeddings object, which can be used to create vector embeddings from text.
    """

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=api_key
    )

    return embeddings

def create_vector_store(chunks, embedding_fn, file_name, vector_store_path="db"):
    """
    Create a vector store from a list of text chunks.

    params:
        chunks (list): A list of generic text chunks to create a vector store from
        embedding_fn (function): A function that takes a string and returns a vector
        file_name (str): The name of the file to associate with the vector store
        vector_store_path (str): The path to save the vector store to (default: "db")

    returns:
        vector_store (Chroma object): A Chroma vector store object
    """

    # create a list of unique ids for each document based on the content to prevent duplicates
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, doc.page_content)) for doc in chunks]

    # ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []

    for chunk, id in zip(chunks, ids):
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)

    # create a new Chroma database from the documents
    vector_store = Chroma.from_documents(documents=unique_chunks,
                                         collection_name=clean_filename(file_name),
                                         embedding=embedding_fn,
                                         ids=list(unique_ids),
                                         persist_directory=vector_store_path)
    
    # the database should save automatically after we create it
    # but we can also force it to save using the persist() method
    vector_store.persist()

    return vector_store

def clean_filename(file_name):
    """
    Remove "(number)" pattern from a filename, because it is not allowed in a collection name

    params:
        file_name (str): The name of the file to clean

    returns:
        cleaned_name (str): The cleaned filename
    """

    # regEx to find "(number)" pattern
    #cleaned_name = re.sub(r'\s\(\d+\)', '', file_name)
    #return cleaned_name
    # Remove invalid characters and replace spaces with underscores
    cleaned_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', file_name)
    # Ensure the name starts and ends with an alphanumeric character
    cleaned_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', cleaned_name)
    # Ensure the name is within the required length
    cleaned_name = cleaned_name[:63]
    # Ensure the name does not contain two consecutive periods
    cleaned_name = re.sub(r'\.\.+', '_', cleaned_name)
    return cleaned_name

def format_docs(docs):
    """
    Format a list of Document objects into a single string.

    params:
        docs (list): A list of Document objects.

    returns:
        A string containing the text of all the documents joined by two newlines.
    """

    return "\n\n".join(doc.page_content for doc in docs)

PROMPT_TEMPLATE = """ 
    You are a recruiter for a company and you are looking for a candidate to fill a position. 
    You have a resume in front of you and you want to know more about the candidate. 
    You want to know the name of the candidate. The resume is attached below. 
    Please provide the name of the candidate.

    If you don't know the answer, say that you don't know. DON'T GUESS OR MAKE UP ANYTHING.

    {context}

    ---

    Answer the question based on the above context: {question}
"""

class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    answer: str = Field(description="Answer to question")
    source: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")

class EducationWithSources(BaseModel):
    """Extracted education information with sources."""
    school: AnswerWithSources
    degree: AnswerWithSources
    major: AnswerWithSources
    location: AnswerWithSources
    duration: AnswerWithSources

class ExperienceWithSources(BaseModel):
    """Extracted experience information with sources."""
    title: AnswerWithSources
    company: AnswerWithSources
    location: AnswerWithSources
    duration: AnswerWithSources
    description: AnswerWithSources
class ExtractedInfoWithSources(BaseModel):
    """Extracted information about the resume:"""
    name: AnswerWithSources  
    email: AnswerWithSources  
    phone: AnswerWithSources  
    education: list[EducationWithSources]  
    work_experience:  list[ExperienceWithSources]
    other_experience: list[ExperienceWithSources]
    skills: AnswerWithSources  
    summary:  AnswerWithSources  

def query_document(vector_store, query, api_key):
    """
    Like a librarian, look up the query in the resume vector store and return a structured response.
    Query a vector store with a question and return a structured response.

    params:
        vector_store (Chroma object): A Chroma vector store object.
        query (str): The question to query the vector store with.
        api_key (str): The OpenAI API key to use when calling the OpenAi Embeddings API.

    returns:
        structured_response_df (pandas DataFrame): A pandas DataFrame with three rows: 'answer', 'source', 'reasoning'
    """
    # create a ChatOpenAI object
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    # create a ChatPromptTemplate object
    retriever = vector_store.as_retriever(search_type="similarity")

    # create a RunnablePassthrough object
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # create a structured response DataFrame
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
    )

    # invoke the RAG chain to get a structured response
    structured_response = rag_chain.invoke(query)
    df = pd.DataFrame([structured_response.dict()])

    
    # transforming into a table with three rows: 'answer', 'source', 'reasoning'
    answer_row = []
    source_row = []
    reasoning_row = []

    seen_education = set()
    seen_work_experience = set()
    seen_other_experience = set()

    for col in df.columns:
        if isinstance(df[col][0], list):
            if col == 'education': 
                education_entries = []
                for entry in df[col][0]:
                    education_entry = (entry['school']['answer'], entry['degree']['answer'], entry['location']['answer'], entry['duration']['answer'])
                    if education_entry not in seen_education:
                        seen_education.add(education_entry)
                        education_entries.append(f"{entry['school']['answer']} | {entry['degree']['answer']} | {entry['location']['answer']} | {entry['duration']['answer']}")
                answer_row.append("\n\n".join(education_entries))

            elif col == 'work_experience':
                experience_entries = []
                for entry in df[col][0]:
                    experience_entry = (entry['title']['answer'], entry['company']['answer'], entry['location']['answer'], entry['duration']['answer'])
                    if experience_entry not in seen_work_experience:
                        seen_work_experience.add(experience_entry)
                        experience_entries.append(f"{entry['title']['answer']} | {entry['company']['answer']} | {entry['location']['answer']} | {entry['duration']['answer']} | {entry['description']['answer']}")
                answer_row.append("\n\n".join(experience_entries))

            elif col == 'other_experience':
                other_experience_entries = []
                for entry in df[col][0]:
                    other_experience_entry = (entry['title']['answer'], entry['company']['answer'], entry['location']['answer'], entry['duration']['answer'])
                    if other_experience_entry not in seen_other_experience:
                        seen_other_experience.add(other_experience_entry)
                        other_experience_entries.append(f"{entry['title']['answer']} | {entry['company']['answer']} | {entry['location']['answer']} | {entry['duration']['answer']} | {entry['description']['answer']}")
                answer_row.append("\n\n".join(other_experience_entries))
        else:
            answer_row.append(df[col][0]['answer'])
            source_row.append(df[col][0]['source'])
            reasoning_row.append(df[col][0]['reasoning'])
    
    # create a DataFrame with three rows: 'answer', 'source', 'reasoning'
    structured_response_df = pd.DataFrame([answer_row, source_row, reasoning_row], columns=df.columns, index=['answer', 'source', 'reasoning']).T
    styled_df = structured_response_df.style.set_properties(**{'height': 'auto', 'width': 'auto'})

    return styled_df


