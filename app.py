from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "jdifbread/my-detector-mini" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 🚨 关键修改：去掉了 torch_dtype=torch.float16，改回 CPU 最兼容的 float32
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, 
    low_cpu_mem_usage=True
)
model.eval()

class TextRequest(BaseModel):
    text: str

@app.post("/detect")
def detect(request: TextRequest):
    try:
        sentences = re.split(r'(?<=[.!?]) +', request.text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        results = []
        total_score = 0
        
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1) # 移除 .float()
            score = round(probs[0][1].item(), 4)
            results.append({"text": sentence, "score": score})
            total_score += score
            
        avg_score = total_score / len(sentences) if sentences else 0

        return {"document_score": round(avg_score, 4), "sentences": results}

    except Exception as e:
        # 如果崩溃了，把错误信息返回给前端，让我们知道到底怎么死的
        print(traceback.format_exc())
        return {"error": str(e), "document_score": 0, "sentences": []}

@app.get("/")
def health():
    return {"status": "ok"}
