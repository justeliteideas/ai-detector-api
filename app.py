from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# 你的 Hugging Face 模型地址
MODEL_ID = "jdifbread/my-detector-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

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
