# rag_core.py
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()




class DocumentLoader:
    """Loads PDFs and TXT files from a folder"""
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir

    def load(self):
        # Load all PDFs
        pdf_loader = PyPDFDirectoryLoader(self.data_dir)
        pdf_docs = pdf_loader.load()

        # Load all TXT files
        txt_loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        txt_docs = txt_loader.load()

        print(f"Loaded {len(pdf_docs)} PDF pages + {len(txt_docs)} TXT files")
        return pdf_docs + txt_docs


class TextChunker:
    """Splits documents into small overlapping chunks"""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def split(self, documents):
        chunks = self.splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks




class VectorStore:
    def __init__(self, persist_dir: str = "faiss_index"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None

    def build(self, chunks):
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        self.vectorstore.save_local(self.persist_dir)          # <--- added
        print(f"FAISS index saved to {self.persist_dir}")
        return self.vectorstore

    def load(self):
        self.vectorstore = FAISS.load_local(
            self.persist_dir,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True   # needed in newer versions
        )
        print(f"Loaded FAISS index with {self.vectorstore.index.ntotal} vectors")
        return self.vectorstore

    def as_retriever(self, k: int = 5):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


class RAGChatbot:
    """Main chatbot class that ties everything together"""
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.loader = DocumentLoader(data_dir)
        self.chunker = TextChunker()
        self.vector_store = VectorStore()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables or secrets")
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",   # fast & cheap on Groq
            temperature=0.1,
            max_tokens=1024
        )
        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
        )

    def build_index(self, force_rebuild: bool = False):
        """One-time indexing step"""
        if not force_rebuild and os.path.exists("chroma_db"):
            self.vector_store.load()
        else:
            docs = self.loader.load()
            chunks = self.chunker.split(docs)
            self.vector_store.build(chunks)

        self.retriever = self.vector_store.as_retriever(k=5)
        print("✅ RAG system is ready!")

    def answer(self, question: str):
        """Get answer for a single question"""
        if self.retriever is None:
            self.build_index()

        # Retrieve relevant chunks
        retrieved_docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate answer
        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return answer

    def chat(self):
        """Interactive chatbot loop"""
        print("\n🤖 RAG Chatbot is live! (type 'exit' or 'quit' to stop)\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not question:
                continue

            answer = self.answer(question)
            print(f"Bot: {answer}\n")



# ====================== RUN THE CHATBOT ======================
if __name__ == "__main__":
    chatbot = RAGChatbot(data_dir="../data")   # change path if needed
    chatbot.build_index(force_rebuild=False)   # set True only first time or when data changes
    chatbot.chat()