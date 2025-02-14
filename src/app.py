import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from agent import get_llm_and_agent


load_dotenv()
def setup_page():
    st.set_page_config(
        page_title="Medical AI",
        page_icon="💬",
        layout="wide"
    )
    st.title("💬 Medical AI")

def setup_sidebar():
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        
        st.header("🤖 Model AI")
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        model_choice = st.radio(
            "Chọn AI Model để trả lời:",
            ["OpenAI GPT-4", "GEMINI 1.5 Flash"],
            index=0 if openai_api_key else 1
        )

        if st.button("Clear Chat"):
            st.session_state.pop("messages", None)
        
        if model_choice == "OpenAI GPT-4" and not openai_api_key:
            st.warning("Không thể chọn OpenAI GPT-4 vì API Key chưa được cung cấp.")
            return None
        elif model_choice == "GEMINI 1.5 Flash" and not google_api_key:
            st.warning("Không thể chọn GEMINI 1.5 Flash vì API Key chưa được cung cấp.")
            return None

        return "gpt4" if model_choice == "OpenAI GPT-4" else "gemini"

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
        ]

def display_chat_history():
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

def handle_user_input(agent_executor, msgs):
    if prompt := st.chat_input("Hãy hỏi tôi bất cứ điều gì về Medical AI!"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        chat_history = st.session_state.messages[:-1]

        with st.chat_message("assistant"):
            try:
                response = agent_executor.invoke(
                    {"input": prompt, "chat_history": chat_history}
                )
                output = response.get("output", "Xin lỗi, tôi không thể trả lời câu hỏi này.")
            except Exception:
                output = f"Hệ thống đang bận, vui lòng thử lại sau."
            
            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)

def main():
    setup_page()
    model_choice = setup_sidebar()
    initialize_chat()
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    display_chat_history()

    agent_executor = get_llm_and_agent(model_choice)
    
    handle_user_input(agent_executor, msgs)

if __name__ == "__main__":
    main()