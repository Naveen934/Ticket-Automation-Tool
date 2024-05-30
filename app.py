from dotenv import load_dotenv
import streamlit as st
from user_utils import *
load_dotenv()
#Creating session variables
if 'HR_tickets' not in st.session_state:
    st.session_state['HR_tickets'] =[]
if 'IT_tickets' not in st.session_state:
    st.session_state['IT_tickets'] =[]
if 'Transport_tickets' not in st.session_state:
    st.session_state['Transport_tickets'] =[]


def main():
    st.header("Automatic Ticket Classification Tool")
    #Capture user input
    st.write("We are here to help you, please ask your question:")
    user_input = st.text_input("🔍")

    with st.sidebar:
        st.title("Please upload your files...📁 ")
        # Upload the pdf file...
        pdf_file= st.file_uploader("Only PDF files allowed", type=["pdf"] ) #, accept_multiple_files=True
        if pdf_file is not None:
            with st.spinner('Wait for it...'):
                    text=read_pdf_data(pdf_file)
                    st.write("👉Reading PDF done")
                    # Create chunks
                    docs_chunks=split_data(text)
                    #st.write(docs_chunks)
                    st.write("👉Splitting data into chunks done")
                    db1=vector_data(docs_chunks)
            st.success("Successfully pushed the embeddings to FIASS") 
    
    if user_input:
        #creating embeddings instance...       
        embeddings=create_embeddings1()
        index=db1
        response=get_answer(index,user_input)
        st.write(response['result'])       
        #Button to create a ticket with respective department
        button = st.button("Submit ticket?")

        if button:
            #Get Response
            embeddings = create_embeddings1()
            query_result = embeddings.embed_query(user_input)
            #loading the ML model, so that we can use it to predit the class to which this compliant belongs to...
            department_value = predict(query_result)
            st.write("your ticket has been sumbitted to : "+department_value)
            #Appending the tickets to below list, so that we can view/use them later on...
            if department_value=="HR":
                st.session_state['HR_tickets'].append(user_input)
            elif department_value=="IT":
                st.session_state['IT_tickets'].append(user_input)
            else:
                st.session_state['Transport_tickets'].append(user_input)



if __name__ == '__main__':
    main()



