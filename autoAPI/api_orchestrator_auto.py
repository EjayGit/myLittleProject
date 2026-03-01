"""
Enhanced API Orchestrator with Auto-Training
Starts training automatically when server starts
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import threading
import queue
import time
from datetime import datetime
import logging
import sys
import os

# Add auto_trainer to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

# Configuration
LLM_API_URL = "http://localhost:5000"
CLEANING_API_URL = "http://localhost:5001"
API_PORT = 8080

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline
training_queue = queue.Queue()

class TrainingPipeline:
    def __init__(self):
        self.active_jobs = {}
    
    def create_job(self, search_query, limit=100, epochs=3):
        job_id = f"job_{int(time.time())}"
        
        job = {
            "id": job_id,
            "status": "created",
            "query": search_query,
            "limit": limit,
            "epochs": epochs,
            "created_at": datetime.now().isoformat()
        }
        
        self.active_jobs[job_id] = job
        training_queue.put(job_id)
        
        logger.info(f"Created job {job_id} for: {search_query}")
        return job
    
    def update_job_status(self, job_id, status):
        if job_id in self.active_jobs:
            self.active_jobs[job_id]["status"] = status
            self.active_jobs[job_id]["updated_at"] = datetime.now().isoformat()

pipeline = TrainingPipeline()

def start_initial_training():
    """Start initial training when server starts"""
    time.sleep(5)  # Wait for services to be ready
    
    logger.info("🚀 Starting initial automated training...")
    
    # Start with a broad topic
    initial_job = pipeline.create_job("artificial intelligence", limit=50, epochs=2)
    
    # Process immediately
    process_job(initial_job["id"])

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "auto_training": "available at http://localhost:8081",
        "services": {
            "llm": "check /api/llm/status",
            "cleaning": "check /api/search?q=test"
        }
    })

@app.route("/api/train_from_search", methods=["POST"])
def train_from_search():
    data = request.get_json() or {}
    query = data.get("query", "science")
    limit = data.get("limit", 50)
    epochs = data.get("epochs", 3)
    
    job = pipeline.create_job(query, limit, epochs)
    
    # Start async processing
    threading.Thread(
        target=process_job,
        args=(job["id"],),
        daemon=True
    ).start()
    
    return jsonify({
        "status": "success",
        "job_id": job["id"],
        "job": job
    })

def process_job(job_id):
    job = pipeline.active_jobs.get(job_id)
    if not job:
        return
    
    pipeline.update_job_status(job_id, "collecting_data")
    
    try:
        # Get data from cleaning service
        clean_url = f"{CLEANING_API_URL}/batch_extract"
        response = requests.get(clean_url, params={
            "q": job["query"],
            "limit": job["limit"],
            "clean_for_llm": "true"
        })
        
        if response.status_code != 200:
            pipeline.update_job_status(job_id, "failed")
            return
        
        clean_data = response.json()
        articles = clean_data.get("articles", [])
        
        # Combine text
        all_text = " ".join([a.get("text", "") for a in articles])
        
        # Send to LLM
        pipeline.update_job_status(job_id, "sending_to_llm")
        
        llm_response = requests.post(
            f"{LLM_API_URL}/set_train_data",
            json={"text": all_text}
        )
        
        if llm_response.status_code != 200:
            pipeline.update_job_status(job_id, "failed")
            return
        
        # Train
        pipeline.update_job_status(job_id, "training")
        
        train_response = requests.post(
            f"{LLM_API_URL}/train",
            json={"epochs": job["epochs"], "batch_size": 32}
        )
        
        pipeline.update_job_status(job_id, "completed")
        logger.info(f"Job {job_id} completed")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        pipeline.update_job_status(job_id, "error")

# ... (rest of the orchestrator endpoints from before)

def process_queue():
    while True:
        try:
            job_id = training_queue.get(timeout=1)
            process_job(job_id)
            training_queue.task_done()
        except queue.Empty:
            continue

if __name__ == "__main__":
    logger.info("Starting LLM Training Pipeline Orchestrator")
    
    # Start queue processor
    threading.Thread(target=process_queue, daemon=True).start()
    
    # Start initial training in background
    threading.Thread(target=start_initial_training, daemon=True).start()
    
    app.run(host="0.0.0.0", port=API_PORT, debug=True)
