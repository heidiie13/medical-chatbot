import os
import json
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from uuid import uuid4

load_dotenv()

def load_data_from_local(filename: str, directory: str) -> dict:
    """
    Hàm đọc dữ liệu từ file JSON data local

    Args:
        filename (str): Tên file JSON
        directory (str): Đường dẫn thư mục chứa file JSON
    Returns:
        dict: Dữ liệu được đọc từ file JSON
    """

    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)

    print(f'Data loaded from {file_path}')
    return data


def seed_milvus(URI_link: str, collection_name: str, filename: str, directory: str, use_gge: bool = False) -> Milvus:
    """
    Hàm seed dữ liệu vào Milvus

    Args:
        URI_link (str): Đường dẫn đến Milvus server
        collection_name (str): Tên của collection trong Milvus
        filename (str): Tên file JSON
        directory (str): Đường dẫn thư mục chứa file JSON
        use_gge (bool): Sử dụng Google Generative AI Embeddings hay không
    Returns:
        Milvus: Đối tượng Milvus đã được seed dữ liệu
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") if use_gge and os.getenv('GOOGLE_API_KEY') else OpenAIEmbeddings(model="text-embedding-3-large")

    local_data = load_data_from_local(filename, directory)

    documents = [
        Document(
            doc['page_content'], 
            metadata={
                'title': doc['metadata'].get('title', 'Default Title'),
                'producer': doc['metadata'].get('producer', 'Unknown'),
                'creator': doc['metadata'].get('creator', 'Unknown'),
                'creationdate': doc['metadata'].get('creationdate', 'Unknown'),
                'moddate': doc['metadata'].get('moddate', 'Unknown'),
                'author': doc['metadata'].get('author', 'Unknown'),
                'source': doc['metadata'].get('source', 'Unknown'),
                'total_pages': doc['metadata'].get('total_pages', 0),
                'page': doc['metadata'].get('page', 0),
                'page_label': doc['metadata'].get('page_label', '1')
            }
        )
        for doc in local_data
    ]

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={'uri': URI_link},
        drop_old=True
    )

    vectorstore.add_documents(documents=documents, ids=uuids)
    return vectorstore

def connect_to_milvus(URI_link: str, collection_name: str, use_gge: bool = False) -> Milvus:
    """
    Hàm kết nối đến collection có sẵn trong Milvus
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection cần kết nối
        use_gge (bool): Sử dụng Google Generative AI Embeddings hay không

    Returns:
        Milvus: Đối tượng Milvus đã được kết nối, sẵn sàng để truy vấn
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") if use_gge and os.getenv('GOOGLE_API_KEY') else OpenAIEmbeddings(model="text-embedding-3-large")

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vectorstore

def main():
    seed_milvus(
        URI_link="http://localhost:19530",
        collection_name="medical_data",
        filename="stack.json",
        directory="data",
        use_gge=True
    )
    print("Done seeding data into Milvus")
    
def test_connect_milvus():
    vector_store = connect_to_milvus(
        URI_link="http://localhost:19530",
        collection_name="medical_data",
        use_gge=True
    )
    retriever= vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    query = "What is AI?"
    print(f"Searching for query: '{query}'")
    results = retriever.get_relevant_documents(query)
    print(f"Đã tìm được {len(results)} tài liệu:")
    for idx, doc in enumerate(results, start=1):
        print(f"Document {idx}: {doc.page_content} (Metadata: {doc.metadata})")
    
if __name__ == "__main__":
    main()
    # test_connect_milvus()