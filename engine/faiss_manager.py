from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from dotenv import load_dotenv
import os
import logging
from typing import List
from utils import get_pdf_files
from docling.document_converter import DocumentConverter



load_dotenv()


KB_FOLDER_DEFAULT = "media/faiss_index"
DATA_FOLDER_DEFAULT = "data"

class KnowledgeBaseManager:
    def __init__(self, kb_folder: str = KB_FOLDER_DEFAULT, data_folder: str = DATA_FOLDER_DEFAULT):
        logging.info("Initializing KnowledgeBaseManager")
        self.kb_folder = kb_folder
        self.data_folder = data_folder
        self.embeddings = OpenAIEmbeddings()
        self.converter = DocumentConverter()


        # Load or create FAISS index
        if os.path.exists(kb_folder) and os.listdir(kb_folder):
            logging.info(f"Loading existing FAISS index from {kb_folder}...")
            self.vectorstore = FAISS.load_local(
                kb_folder, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            self._initialize_from_pdfs()

    def _chunk_pdf(self, path):
        logging.info(f"Chunking PDF: {path}")
        result = self.converter.convert(path)
        parsed_data = []
        current_chunk = ""

        for element in result.assembled.elements:
            
            element_type = element.label.name
            text_content = element.text or ""

            if element_type in ['PAGE_HEADER', 'PAGE_FOOTER', 'DOCUMENT_INDEX', "PICTURE"]:
                continue
            
            if element_type == "SECTION_HEADER":
                if current_chunk: # Only append if the chunk has content
                    parsed_data.append(current_chunk)
                
                current_chunk = "" # Start the new chunk
                
            else:
                if text_content and len(text_content.split()) >= 3:
                    current_chunk += text_content + "\n"
                

        if current_chunk:
            parsed_data.append(current_chunk.strip())
        
        logging.info(f"Finished chunking PDF: {path}, {len(parsed_data)} chunks created.")
        return parsed_data
    

    def _initialize_from_pdfs(self):
        logging.info("Creating new FAISS index from PDFs...")
        os.makedirs(self.kb_folder, exist_ok=True)
        
        pdf_files = get_pdf_files(self.data_folder)
        if not pdf_files:
            logging.warning("No PDF files found in the data folder.")
            # Create an empty index
            dummy_doc = Document(page_content="init", metadata={"id": "dummy"})
            self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            self.vectorstore.save_local(self.kb_folder)
            logging.info("Empty FAISS index created.")
            return

        for pdf_file in pdf_files:
            chunked_pdf = self._chunk_pdf(pdf_file) 
            for chunk in chunked_pdf:
                self.add_text_data(chunk)

        self.vectorstore.save_local(self.kb_folder)
        logging.info(f"Added documents and saved index to {self.kb_folder}")

    def add_text_data(self, text: str, chunk_size: int = 2000, chunk_overlap: int = 400):
        """
        Chunks text, embeds it, adds to vectorstore, and saves locally.
        """
        logging.info("Adding text data to vectorstore.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        

        docs = text_splitter.create_documents([text])
        logging.info(f"Split text into {len(docs)} chunks.")


        if hasattr(self, 'vectorstore'):
            self.vectorstore.add_documents(docs)
        else:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)


    def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 20) -> List[Document]:
        """
        Retrieves documents and re-ranks them using a Cross-Encoder.
        
        Args:
            query: The user query.
            top_k: The final number of documents to return.
            fetch_k: The number of documents to initially fetch from FAISS (candidates).
        """
        logging.info(f"Retrieving documents for query: {query}")
        #TODO: MAYBE ADD RERANKER
        results_with_scores = self.vectorstore.similarity_search_with_relevance_scores(query, k=fetch_k)
        logging.info(f"Retrieved {len(results_with_scores)} documents.")
        return results_with_scores

    def get_all_documents(self) -> List[Document]:
        """
        Retrieves all documents from the vectorstore.
        """
        logging.info("Retrieving all documents from vectorstore.")
        return self.vectorstore.similarity_search(" ", k=1000)

# Example Usage
# if __name__ == "__main__":
#     kb = KnowledgeBaseManager()
    
#     # 1. Add Data (This will be done automatically if PDFs are present)
    
#     # 2. Retrieve with Rerank
#     docs = kb.retrieve_with_rerank("What is LangChain?")
    
#     if docs:
#         print("\n--- Top Result ---")
#         top_result = docs[0]
#         top_content = top_result[0].page_content[:50]
#         top_score = top_result[1]
#         print(f"Score: {top_score:.4f} | Content: {top_content}...")
    
#     # 3. Get all documents
#     all_docs = kb.get_all_documents()
#     print(f"\n--- All Documents ({len(all_docs)}) ---")
#     for doc in all_docs:
#         print(doc.page_content[:100] + "...")