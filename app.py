from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = FastAPI()

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
    # 1. 真实切分：按句号、问号、感叹号把文章切成句子
    sentences = re.split(r'(?<=[.!?]) +', request.text)
    
    results = []
    total_score = 0
    count = 0

    # 2. 逐句分析
    for sentence in sentences:
        if not sentence.strip(): continue
        
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits.float(), dim=-1)
        ai_prob = probs[0][1].item()
        
        # 如果分数太极端，稍微平滑一下（可选，为了展示效果）
        ai_prob = round(ai_prob, 4)
        
        results.append({
            "text": sentence,
            "score": ai_prob
        })
        total_score += ai_prob
        count += 1

    # 3. 计算整篇真实平均分
    avg_score = total_score / count if count > 0 else 0

    return {
        "document_score": round(avg_score, 4),
        "sentences": results
    }

@app.get("/")
def health():
    return {"status": "ok"}
