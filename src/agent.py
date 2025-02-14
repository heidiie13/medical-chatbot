import os
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent 
from seed_data import connect_to_milvus
from prompt import system_prompt

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_retriever(URI_link: str = 'http://localhost:19530', collection_name: str = "medical_data", use_gge: bool = False) -> EnsembleRetriever:
    """
    Tạo một ensemble retriever kết hợp giữa Milvus và BM25
    Args:
        collection_name (str): Tên của collection trong Milvus
        use_gge (bool): Sử dụng Google Generative AI Embeddings hay không
    """
    try:
        if use_gge:
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
        else:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        vectorstore = connect_to_milvus(URI_link, collection_name, use_gge)
        milvus_retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("", k=50)
        ]
        
        if not documents:
            documents = [
                Document(
                    page_content="Không tìm thấy tài liệu nào.",
                    metadata={"source": "none"}
                )
            ]
            
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        ensemble_retriever = EnsembleRetriever(
            retrievers=[milvus_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        return ensemble_retriever
        
    except Exception as e:
        print(f"Lỗi khi khởi tạo retriever: {str(e)}")
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)


def get_llm_and_agent(model_choice: str) -> AgentExecutor:
    """
    Khởi tạo Language Model và Agent với cấu hình cụ thể
    Args:
        model_choice: Lựa chọn model ("gpt4" hoặc "gemini")
    """

    if model_choice == "gpt4":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        llm = ChatOpenAI(
            temperature=0,
            streaming=True,
            model='gpt-4')
        retriever = get_retriever()
    elif model_choice == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        llm = ChatGoogleGenerativeAI(
            temperature=0, 
            streaming=True, 
            model='gemini-1.5-flash')
        retriever = get_retriever(use_gge=True)
    else:
        return None
    
    medical_retriever_tool = create_retriever_tool(
        retriever,
        name="search_medical_data",
        description="""
        Dùng để tìm kiếm thông tin sức khỏe, bệnh tật, triệu chứng y tế từ cơ sở dữ liệu. 
        Input: Câu truy vấn tìm kiếm về thông tin y tế
        Output: Danh sách các tài liệu y tế liên quan
        Khi nào sử dụng: Khi cần tra cứu thông tin y tế cụ thể
        """
        )
    
    tools = [medical_retriever_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3, early_stopping_method="generate", handle_parsing_errors=True)

def test_get_llm_and_agent(model_choice: str = "gpt4"):
    """Test hàm get_llm_and_agent với lựa chọn model."""
    try:
        agent_executor = get_llm_and_agent(model_choice)
        if agent_executor is None:
            print("get_llm_and_agent trả về None. Kiểm tra lại cấu hình API key.")
        else:
            query = "Bạn có thể tìm kiếm một số thông tin y tế về bệnh hô hấp được không?"
            print(f"Running agent với query: '{query}'")
            response = agent_executor.invoke({"input":query})
            print("Agent response:", response)
    except Exception as e:
        print(f"Lỗi khi test get_llm_and_agent: {e}")

if __name__ == "__main__":
    test_get_llm_and_agent("gemini")
