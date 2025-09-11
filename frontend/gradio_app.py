# -------- Gradio app --------
from config.settings import K, SYSTEM_PROMPT
from llm.inference import call_llm, coaching_wrapper
from llm.prompt_builder import build_messages
from rag.retrieve import retrieve
import gradio as gr


def create_app(index, texts, metas, embed_model):
    def respond(user_message, history):
        if history is None:
            history = []

        # Lưu tin nhắn user
        history.append({"role": "user", "content": user_message})

        # Tạo tuple_history cho build_messages
        tuple_history = []
        for i in range(0, len(history) - 1, 2):  # duyệt cặp user - assistant
            user_msg = history[i]["content"]
            assistant_msg = history[i + 1]["content"] if i + 1 < len(history) else ""
            tuple_history.append((user_msg, assistant_msg))

        # RAG retrieve
        retrieved = retrieve(user_message, index, texts, metas, embed_model, k=K)

        # Build messages cho LLM
        messages = build_messages(SYSTEM_PROMPT, tuple_history, user_message, retrieved)

        # Gọi model HuggingFace
        assistant_reply = call_llm(messages)
        assistant_reply = coaching_wrapper(assistant_reply, user_message)

        # Lưu tin nhắn assistant
        history.append({"role": "assistant", "content": assistant_reply})

        # Trả về cho textbox, state, và chatbot (đều cùng history)
        return "", history, history


    with gr.Blocks() as demo:
        gr.Markdown("## Life Coach (RAG + Local HuggingFace Model)\nNhập câu hỏi của bạn bên dưới.")
        chatbot = gr.Chatbot([], elem_id="chatbot", height=500, type="messages")
        state = gr.State([])
        msg = gr.Textbox(show_label=False, placeholder="Nhập tin nhắn và Enter để gửi...")
        msg.submit(respond, [msg, state], [msg, state, chatbot])
        demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)