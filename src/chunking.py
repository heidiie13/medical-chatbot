import os
import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf_documents(directory: str) -> list:
    """
    Hàm tải dữ liệu từ thư mục chứa các file pdf
    Args:
        directory_path (str): Đường dẫn đến thư mục chứa các file pdf
    Returns:
        list: Danh sách các Document object đã được chia nhỏ thành các chunk
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f'Directory {directory} not found')
    
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    return chunks

def save_data_loader(documents, filename, directory):
    """
    Lưu danh sách documents vào file JSON
    Args:
        documents (list): Danh sách các Document object cần lưu
        filename (str): Tên file JSON
        directory (str): Đường dẫn thư mục lưu file
    Returns:
        None: Hàm không trả về giá trị, chỉ lưu file và in thông báo
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)

    data_to_save = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]

    with open(file_path, 'w') as file:
        json.dump(data_to_save, file, indent=2)
        
    print(f'Data saved to {file_path}')

def main():
    medical_data_dir = os.path.join("..", "medical_data")    
    data = load_pdf_documents(medical_data_dir)
    print("len(data):", len(data))
    save_data_loader(data, 'stack.json', 'data')

if __name__ == "__main__":
    main()