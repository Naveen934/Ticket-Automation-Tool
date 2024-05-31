import streamlit as st
from user_utils import *


def main():
    
    st.set_page_config(page_title="Dump PDF to FIASS- Vector Store")
    st.title("Please upload your files...📁 ")
    # Upload the pdf file...
    pdf_file= st.file_uploader("Only PDF files allowed", type=["pdf"], accept_multiple_files=True ) #, accept_multiple_files=True
    if pdf_file :
        with st.spinner('Wait for it...'):
                text1=read_pdf_data(pdf_file)
                st.write("👉Reading PDF done")
                # Create chunks
                docs_chunks=split_data(text1)
                #st.write(docs_chunks)
                st.write("👉Splitting data into chunks done")
                index=create_vectorDB(docs_chunks)
        st.success("Successfully pushed the embeddings to FIASS") 

if __name__ == '__main__':
    main()