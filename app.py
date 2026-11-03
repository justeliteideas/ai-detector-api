from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# 你的 Hugging Face 模型地址
MODEL_ID = "jdifbread/my-detector-model"

print("正在下载并加载模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 🚨 关键：加上 low_cpu_mem_usage=True，极度节省内存
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, 
    low_cpu_mem_usage=True
)
model.eval()
print("模型加载完毕！")

class TextRequest(BaseModel):
    text: str

@app.post("/detect")
def detect(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_prob = probs[0][1].item()
    return {"ai_probability": round(ai_prob, 4)}

@app.get("/")
def health():
    return {"status": "ok"}
