import streamlit as st
import base64
from functions import *

# initialize API key in session 
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

def display_pdf(uploaded_file):
    """
    Display a PDF file that has been uploaded to the app.

    The PDF will be displayed in an iframe, with the width and height set to 700x1000 pixels.

    Parameters
    ----------
    uploaded_file : streamlit.uploaded_file_manager.UploadedFile
        The PDF file that has been uploaded to the app.

    Returns
    -------
    None
    """

    # read file as bytes
    bytes_data = uploaded_file.getvalue()

    # convert to base64
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')

    # embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # display PDF
    st.markdown(pdf_display, unsafe_allow_html=True)

def load_streamlit_page():
    """
    Load the streamlit page with 2 columns. Left column contains a text input box for the user to input their OpenAI API Key, 
    and a file uploader for the user to upload a PDF resume. Right column contains a header and text that greet the user
    and explain the purpose of the tool.

    Returns
    -------
        col1: The left column Streamlit object
        col2: The right column Streamlit object
        uploaded_file: The uploaded PDF file
    """
    
    st.set_page_config(layout="wide", page_title="Resume Analyzer")

    # design page layout with 2 columns: file uploader on the left, and other interactions on the right
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.header("Input your OpenAI API key")
        st.text_input('OpenAI API Key', type="password", key="api_key",
                      label_visibility="collapsed", disabled=False)
        st.header("Upload file")
        uploaded_file = st.file_uploader("Upload your resume", type="pdf")

    return col1, col2, uploaded_file

# load streamlit page
col1, col2, uploaded_file = load_streamlit_page()

# process the input
if uploaded_file is not None:
    with col2:
        display_pdf(uploaded_file)

        # load the documents
        documents = get_pdf_text(uploaded_file)
        st.session_state.vector_store = create_vector_store_from_texts(documents, 
                                                                       api_key=st.session_state.api_key, 
                                                                       file_name=uploaded_file.name)
        
        st.write("Resume loaded successfully!")
    
    # generate answer
    with col1:
        if (st.button("Generate table")):
            with st.spinner("Generating answer"):
                # load vector_store
                answer = query_document(vector_store=st.session_state.vector_store,
                                        query="Give me the name, email, phone number, and education of the person in this resume.",
                                        api_key=st.session_state.api_key)
                placeholder = st.empty()
                placeholder.write(answer)