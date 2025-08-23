import random as rd
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.settings import MODEL_NAME
from llama_cpp import Llama

# -------- Load HuggingFace model --------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model {MODEL_NAME} on {device}...")

# HuggingFace model và tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#     device_map="auto"
# )

# Chuyển sang gguf model


# Load model llama.cpp từ file .gguf
from llama_cpp import Llama

llm = Llama(
    model_path=MODEL_NAME,
    n_ctx=2048,   # hoặc 4096 nếu model hỗ trợ
    n_threads=8
)


# Hàm call_llm thay thế
def call_llm(messages, max_tokens=256, temperature=0.7):
    # Chuyển messages thành prompt text
    prompt = ""
    for m in messages:
        if m["role"] == "system":
            prompt += f"{m['content']}\n"
        elif m["role"] == "user":
            prompt += f"### User: {m['content']}\n"
        elif m["role"] == "assistant":
            prompt += f"### Assistant: {m['content']}\n"
    prompt += "### Assistant:"

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["### User:"]
    )
    response = output["choices"][0]["text"].strip()
    return response


# def call_llm(messages, max_tokens=512, temperature=0.7):
#     # Chuyển messages thành prompt text
#     prompt = ""
#     for m in messages:
#         if m["role"] == "system":
#             prompt += f"{m['content']}\n"
#         elif m["role"] == "user":
#             prompt += f"### User: {m['content']}\n"
#         elif m["role"] == "assistant":
#             prompt += f"### Assistant: {m['content']}\n"
#     prompt += "### Assistant:"

#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=max_tokens,
#         temperature=temperature,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response.split("### Assistant:")[-1].strip()


import re, random as rd

def coaching_wrapper(reply, user_message):
    openers = [
        "Bạn có thể chia sẻ thêm điều gì khiến bạn cảm thấy như vậy không?",
        "Điều gì làm bạn lo lắng nhất trong chủ đề này?",
        "Bạn mong muốn thay đổi điều gì trước tiên?",
        "Bạn nghĩ nguyên nhân chính đến từ đâu?",
    ]
    reply = reply.strip()

    # Tách câu theo dấu ?
    sentences = re.split(r'(?<=\?)', reply)

    # Nếu có nhiều câu hỏi, chỉ giữ câu hỏi cuối cùng
    if len(sentences) > 1:
        non_questions = [s for s in sentences if "?" not in s]
        questions = [s for s in sentences if "?" in s]

        reply = " ".join(non_questions).strip()
        if questions:
            reply += " " + questions[-1].strip()

    # Nếu không có câu hỏi nào, thêm 1 câu hỏi ngẫu nhiên
    if "?" not in reply:
        reply += " " + rd.choice(openers)

    return reply
