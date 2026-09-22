"""FastAPI web server for Fine-Tuning powered by Google Gemini 2.5 Flash."""
import os
import sys
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fine_tune_pipeline import FineTunePipeline

app = FastAPI(title="Fine-Tuning AI Studio", version="1.0.0")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_file = os.path.join(base_dir, "data", "domain_data.jsonl")

pipeline = FineTunePipeline(data_file)

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_file = os.path.join(static_dir, "index.html")
    if os.path.exists(html_file):
        return FileResponse(html_file)
    return HTMLResponse("<h2>Fine-Tuning Web App</h2>")

@app.get("/api/run-fine-tune")
@app.post("/api/run-fine-tune")
def run_fine_tune_api():
    train_res = pipeline.train_lora_adapter(num_epochs=3)
    eval_res = pipeline.evaluate_base_vs_adapted()
    return {"training": train_res, "evaluation": eval_res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
