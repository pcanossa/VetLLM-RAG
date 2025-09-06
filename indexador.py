# 1. Importar bibliotecas
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# 2. Definir pastas e modelo de embedding
DIRETORIO_PDFS = os.getenv("DIRETORIO_PDFS")
DIRETORIO_DB = os.getenv("DIRETORIO_DB")
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 3. Carregar os documentos
loader = PyPDFDirectoryLoader(DIRETORIO_PDFS)
documentos = loader.load()
print(f"Carregados {len(documentos)} documentos.")

# 4. Dividir os documentos em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documentos)
print(f"Documentos divididos em {len(chunks)} chunks.")

# 5. Inicializar o modelo de embeddings
embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING)

# 6. Criar o banco de dados vetorial e salvar
# O ChromaDB irá processar os chunks, gerar os embeddings e salvar no disco
db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=DIRETORIO_DB
)

print("Banco de dados vetorial criado e salvo com sucesso!")
