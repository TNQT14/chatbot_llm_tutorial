import random as rd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.settings import MODEL_NAME
# -------- Load HuggingFace model --------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
def call_llm(messages, max_tokens=512, temperature=0.7):
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
    return response.split("### Assistant:")[-1].strip()

def coaching_wrapper(reply, user_message):
    # thêm câu mở rộng gợi mở
    openers = [
        "Bạn có thể chia sẻ thêm điều gì khiến bạn cảm thấy như vậy không?",
        "Điều gì làm bạn lo lắng nhất trong chủ đề này?",
        "Bạn mong muốn thay đổi điều gì trước tiên?",
        "Bạn nghĩ nguyên nhân chính đến từ đâu?",
    ]
    reply = reply.strip()
    reply += " " + rd.choice(openers)

    return reply
