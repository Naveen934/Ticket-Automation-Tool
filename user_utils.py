from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from dotenv import load_dotenv


load_dotenv()


llm=HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct",max_length=128)


#**********Functions to help you load documents to PINECONE************

#Read PDF data
def read_pdf_data(pdf_file):
    text=""
    for pdf in pdf_file:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def read_pdf_data1(pdf_file):
    loader=PyPDFDirectoryLoader(pdf_file) ## Data Ingestion
    text=loader.load()
    return text

def read_pdf_data2(pdf_file):
    loader=PyPDFDirectoryLoader(pdf_file) ## Data Ingestion
    documents=loader.load()
    text = "\n".join(doc.page_content for doc in documents)
    return text

#Split data into chunks
def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_text(text)
    docs_chunks =text_splitter.create_documents(docs)
    return docs_chunks

#Create embeddings instance

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

#Function to push data to FAISS

def create_vectorDB(docs):

    db=FAISS.from_documents(docs,embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")) 
    db.save_local("faiss_index")

def get_DB(embeddings):
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
   
    return new_db


def get_answer(db,user_input):
    qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=db.as_retriever() )
    
    result = qa_chain({"query":user_input })
    return result
def predict(query_result):
    Fitmodel = joblib.load('modelsvm.pk1')
    result=Fitmodel.predict([query_result])
    return result[0]


#*********Functions for dealing with Model related tasks...************

#Read dataset for model creation
def read_data(data):
    df = pd.read_csv(data,delimiter=',', header=None)  
    return df

#Generating embeddings for our input dataset
def create_embeddings(df,embeddings):
    df[2] = df[0].apply(lambda x: embeddings.embed_query(x))
    return df

#Splitting the data into train & test
def split_train_test__data(df_sample):
    # Split into training and testing sets
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
    list(df_sample[2]), list(df_sample[1]), test_size=0.25, random_state=0)
    print(len(sentences_train))
    return sentences_train, sentences_test, labels_train, labels_test

#Get the accuracy score on test data
def get_score(svm_classifier,sentences_test,labels_test):
    score = svm_classifier.score(sentences_test, labels_test)
    return score
