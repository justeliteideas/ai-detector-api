from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = FastAPI()

# 🚨 允许跨域请求，防止 Z.ai 报跨域错误
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "jdifbread/my-detector-mini" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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
    # 按句号、问号、感叹号切分句子
    sentences = re.split(r'(?<=[.!?]) +', request.text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return {"document_score": 0, "sentences": []}

    # 🚨 关键升级：批量处理，一次性算完所有句子！
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    probs = torch.nn.functional.softmax(outputs.logits.float(), dim=-1)
    ai_probs = probs[:, 1].tolist()
    
    results = []
    total_score = 0
    
    for i, sentence in enumerate(sentences):
        score = round(ai_probs[i], 4)
        results.append({"text": sentence, "score": score})
        total_score += score
        
    avg_score = total_score / len(sentences) if sentences else 0

    return {
        "document_score": round(avg_score, 4),
        "sentences": results
    }

@app.get("/")
def health():
    return {"status": "ok"}
