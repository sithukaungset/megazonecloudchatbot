import streamlit as st
import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

def format_polygon(polygon):
    if not polygon:
        return "N/A"
    return ", ".join(["[{}, {}]".format(p.x, p.y) for p in polygon])

def analyze_general_documents(pdf_stream):
    document_analysis_client = DocumentAnalysisClient(
        endpoint="https://formtestlsw.cognitiveservices.azure.com", 
        credential=AzureKeyCredential("2fe1b91a80f94bb2a751f7880f00adf6")
    )

    with open("temp_pdf_for_analysis.pdf", "wb") as temp_file:
        temp_file.write(pdf_stream.read())

    with open("temp_pdf_for_analysis.pdf", "rb") as temp_file:
        poller = document_analysis_client.begin_analyze_document("prebuilt-document", temp_file)
        result = poller.result()
    
    # Displaying Key-Value Pairs
    st.subheader("Key-value pairs found in document")
    for kv_pair in result.key_value_pairs:
        key_text = kv_pair.key.content if kv_pair.key else "N/A"
        value_text = kv_pair.value.content if kv_pair.value else "N/A"
        st.write(f"Key: {key_text} - Value: {value_text}")
    
    # Displaying lines of text
    st.subheader("Text content by page")
    for page in result.pages:
        st.write(f"--- Page {page.page_number} ---")
        for line in page.lines:
            st.write(line.content)

    # Displaying tables (just a basic example)
    st.subheader("Tables in the document")
    for table_idx, table in enumerate(result.tables):
        st.write(f"--- Table {table_idx + 1} ---")
        for cell in table.cells:
            st.write(f"Row {cell.row_index}, Column {cell.column_index}: {cell.content}")

def main():
    st.title("Azure Form Recognizer with Streamlit")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        if st.button("Analyze"):
            with st.spinner("Analyzing the PDF..."):
                analyze_general_documents(uploaded_file)
        
if __name__ == "__main__":
    main()
