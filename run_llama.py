from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model và tokenizer
model = AutoModelForCausalLM.from_pretrained("TNQT14/Llama-3.2-1B-Instruct-Chat-sft")
tokenizer = AutoTokenizer.from_pretrained("TNQT14/Llama-3.2-1B-Instruct-Chat-sft")

# Prompt nhập vai Life Coach
input_text = """
Bạn là một Life Coach tận tâm, luôn lắng nghe, thấu hiểu và hỗ trợ khách hàng tìm ra giải pháp tích cực trong cuộc sống.

### User: Xin chào! Hôm nay tôi cảm thấy hơi buồn vì code của tôi không chạy đúng. Tôi đã cố gắng sửa nhưng không thành công.
### Assistant:"""

# Mã hóa đầu vào
inputs = tokenizer(input_text, return_tensors="pt")

# Sinh câu trả lời
outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)

# Giải mã và in kết quả
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("### Assistant:")[-1].strip())