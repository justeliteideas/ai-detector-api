from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# 🚨 确认这里是你 Hugging Face 的迷你模型地址
MODEL_ID = "jdifbread/my-detector-mini" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 🚨 极限省内存模式
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, 
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
model.eval()

class TextRequest(BaseModel):
    text: str

@app.post("/detect")
def detect(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits.float(), dim=-1)
    ai_prob = probs[0][1].item()
    return {"ai_probability": round(ai_prob, 4)}

@app.get("/")
def health():
    return {"status": "ok"}
