import re
import random as rd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.settings import MODEL_NAME

# -------- Detect device (MPS / CUDA / CPU) --------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")


# -------- Load HuggingFace model --------
# device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model {MODEL_NAME} on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32 if device != "cpu" else torch.float32,
    device_map={"": device}
)

# Hàm gọi LLM
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

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Lấy phần sau "### Assistant:"
    response = response.split("### Assistant:")[-1].strip()
    # Loại bỏ ký tự lặp quá mức
    response = re.sub(r'(.)\1{5,}', r'\1', response)
    # Chuẩn hóa whitespace
    response = re.sub(r'\s+', ' ', response).strip()
    return response


# Hàm coaching wrapper
def coaching_wrapper(reply, user_message):
    openers = [
        "Bạn có thể chia sẻ thêm điều gì khiến bạn cảm thấy như vậy không?",
        "Điều gì làm bạn lo lắng nhất trong chủ đề này?",
        "Bạn mong muốn thay đổi điều gì trước tiên?",
        "Bạn nghĩ nguyên nhân chính đến từ đâu?",
    ]
    reply = reply.strip()

    # Tách câu
    sentences = re.split(r'(?<=[?!.])', reply)

    # Nếu có nhiều câu hỏi, giữ câu hỏi cuối
    if len(sentences) > 1:
        non_questions = [s for s in sentences if "?" not in s]
        questions = [s for s in sentences if "?" in s]

        reply = " ".join(non_questions).strip()
        if questions:
            reply += " " + questions[-1].strip()

    # Nếu không có câu hỏi nào, thêm 1 câu hỏi mở rộng
    if "?" not in reply:
        reply += " " + rd.choice(openers)

    return reply
