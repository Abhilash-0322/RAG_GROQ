import os
import json
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM and prompt
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a CSV assistant. Your task is to retrieve the correct translation for a given question based on the context.
    Respond in JSON format with the following structure:
    {{
        "translation": "<retrieved_translation>"
    }}
    If you cannot find the translation, respond with "I cannot find the translation."
    <context>
    {context}
    <context>
    Question: {input}
    """
)

st.title("Bhagavad Gita Verses Evaluator")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

# Global variables
data = None
vectors = None


def process_csv(csv_file):
    """
    Processes the CSV file and creates vector embeddings for questions and translations.
    """
    df = pd.read_csv(csv_file)

    # Validate columns
    if "question" not in df.columns or "translation" not in df.columns:
        st.error("CSV must contain 'question' and 'translation' columns.")
        return None, None

    # Create Documents
    documents = [
        Document(
            page_content=f"Question: {row['question']}\nTranslation: {row['translation']}",
            metadata={"row_index": i}
        )
        for i, row in df.iterrows()
    ]

    # Split and embed documents
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)
    vectors = FAISS.from_documents(final_documents, embeddings)

    return df, vectors


def evaluate_rag(df, vectors):
    """
    Evaluates the RAG performance by comparing retrieved translations with the expected translations.
    """
    sn=0
    correct_answers = 0
    # total_questions = len(df)
    total_questions = 50

    # Create chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    results = []

    # Loop through questions
    for idx, row in df.iterrows():
        if sn>50:
            break
        sn+=1
        question = row['question']
        expected_translation = row['translation']

        # Get response
        response = retrieval_chain.invoke({'input': question})

        try:
            response_data = json.loads(response['answer'])
            retrieved_translation = response_data.get("translation", "")

            # Compare translations
            is_correct = expected_translation.strip().lower() == retrieved_translation.strip().lower()
            if is_correct:
                correct_answers += 1

            results.append({
                "question": question,
                "expected_translation": expected_translation,
                "retrieved_translation": retrieved_translation,
                "is_correct": is_correct
            })
        except (json.JSONDecodeError, KeyError):
            results.append({
                "question": question,
                "expected_translation": expected_translation,
                "retrieved_translation": "Error in response",
                "is_correct": False
            })

    # Calculate accuracy
    accuracy = (correct_answers / total_questions) * 100
    return accuracy, results


# Step 1: Process the CSV and embed
if st.button("Process CSV") and uploaded_file:
    data, vectors = process_csv(uploaded_file)
    if vectors:
        st.success("CSV processed and embeddings created successfully.")

# Step 2: Evaluate RAG
if st.button("Evaluate RAG") and data is not None and vectors is not None:
    accuracy, results = evaluate_rag(data, vectors)
    st.write(f"**Accuracy**: {accuracy:.2f}%")
    st.write(f"**Correct Answers**: {sum(result['is_correct'] for result in results)}")
    st.write(f"**Total Questions**: {len(results)}")

    # Display results
    st.subheader("Evaluation Results")
    for result in results:
        st.write(f"**Question**: {result['question']}")
        st.write(f"**Expected Translation**: {result['expected_translation']}")
        st.write(f"**Retrieved Translation**: {result['retrieved_translation']}")
        st.write(f"**Correct**: {result['is_correct']}")
        st.write("---")
elif uploaded_file is None:
    st.warning("Please upload a CSV file before evaluating.")
elif data is None or vectors is None:
    st.warning("Please process the CSV before evaluating.")
