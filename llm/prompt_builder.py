# -------- prompt building & local LLM call --------
from config.settings import MAX_TURNS_HISTORY, TGROW_INSTRUCTION


def build_messages(system_prompt, history_tuples, user_question, retrieved_docs):
    messages = [{"role": "system", "content": system_prompt + "\n\n" + TGROW_INSTRUCTION}]

    if retrieved_docs:
        ctx = "\n\n---\n".join([f"Source: {d['source']}\n{d['text']}" for d in retrieved_docs])
        messages.append({"role": "system", "content": "Tài liệu tham khảo:\n\n" + ctx})

    for user_text, assistant_text in history_tuples[-MAX_TURNS_HISTORY:]:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": user_question})
    return messages