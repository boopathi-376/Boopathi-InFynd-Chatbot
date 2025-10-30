import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====================== CONFIG ======================
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
LOCAL_DIR = "C:/OPENBOT/local_llm/Phi-3-mini-4k-instruct"

# ====================== DEVICE SETUP ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ====================== LOAD OR DOWNLOAD ======================
try:
    print(f"📦 Checking for local model in: {LOCAL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_DIR,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    print("✅ Loaded model from local folder.")
except Exception as e:
    print(f"🌐 Local model not found. Downloading from Hugging Face: {MODEL_NAME}")
    print(f"⚠️ Reason: {e}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    model.save_pretrained(LOCAL_DIR)
    tokenizer.save_pretrained(LOCAL_DIR)
    print(f"✅ Model downloaded and saved locally at: {LOCAL_DIR}")

# ====================== TEST PROMPT ======================
prompt = "Explain what a vector database is in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\n⚙️ Generating response...")
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, top_p=0.9)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n🧠 Model Output:\n")
print(response)
