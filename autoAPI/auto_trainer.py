"""
Auto Trainer - Automated training pipeline
Searches for topics, collects data, and trains the LLM automatically
"""

import requests
import time
import threading
import schedule
from datetime import datetime
import random
import logging
from typing import List, Dict, Optional
from flask import request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoTrainer:
    """Automated training pipeline manager"""
    
    def __init__(self, orchestrator_url="http://localhost:8080"):
        self.orchestrator_url = orchestrator_url
        self.running = False
        self.current_job = None
        self.training_history = []
        
        # Training topics (can be expanded)
        self.training_topics = [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural networks",
            "natural language processing",
            "computer science",
            "mathematics",
            "physics",
            "biology",
            "chemistry",
            "engineering",
            "technology",
            "programming",
            "algorithms",
            "data science",
            "robotics",
            "quantum computing",
            "neuroscience",
            "psychology",
            "philosophy",
            "economics",
            "history of science",
            "scientific method",
            "logic",
            "statistics"
        ]
        
        # Training configurations
        self.training_configs = [
            {"limit": 30, "epochs": 2, "batch_size": 32},  # Quick training
            {"limit": 50, "epochs": 3, "batch_size": 32},  # Standard training
            {"limit": 100, "epochs": 4, "batch_size": 64}, # Extended training
        ]
    
    def check_services(self) -> bool:
        """Check if all services are running"""
        try:
            response = requests.get(f"{self.orchestrator_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ All services are healthy")
                return True
            else:
                logger.error("❌ Services health check failed")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to orchestrator: {e}")
            return False
    
    def start_auto_training(self, 
                           topics: Optional[List[str]] = None,
                           interval_minutes: int = 30,
                           immediate: bool = True):
        """Start automated training pipeline"""
        
        if not self.check_services():
            logger.error("❌ Cannot start training - services not available")
            return False
        
        self.running = True
        logger.info(f"🚀 Starting auto-training pipeline (interval: {interval_minutes} minutes)")
        
        # Schedule training jobs
        if immediate:
            # Start first training immediately
            self.run_training_cycle()
        
        # Schedule subsequent trainings
        schedule.every(interval_minutes).minutes.do(self.run_training_cycle)
        
        # Start scheduler in background thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("✅ Auto-trainer started successfully")
        return True
    
    def _run_scheduler(self):
        """Run the scheduler in background"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def run_training_cycle(self):
        """Run one complete training cycle"""
        if not self.running:
            return
        
        try:
            # Select random topic and config
            topic = random.choice(self.training_topics)
            config = random.choice(self.training_configs)
            
            logger.info(f"📚 Starting training cycle: {topic}")
            logger.info(f"   Config: {config['limit']} articles, {config['epochs']} epochs")
            
            # Start training job
            job_response = requests.post(
                f"{self.orchestrator_url}/api/train_from_search",
                json={
                    "query": topic,
                    "limit": config["limit"],
                    "epochs": config["epochs"],
                    "batch_size": config["batch_size"]
                },
                timeout=10
            )
            
            if job_response.status_code != 200:
                logger.error(f"❌ Failed to start training job: {job_response.text}")
                return
            
            job_data = job_response.json()
            job_id = job_data["job_id"]
            self.current_job = job_id
            
            logger.info(f"   Job ID: {job_id}")
            
            # Monitor job progress
            self._monitor_job(job_id, topic, config)
            
            # Record in history
            self.training_history.append({
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "job_id": job_id,
                "config": config,
                "status": "completed"
            })
            
            # Clean up old history
            if len(self.training_history) > 100:
                self.training_history = self.training_history[-50:]
                
        except Exception as e:
            logger.error(f"❌ Training cycle failed: {e}")
    
    def _monitor_job(self, job_id: str, topic: str, config: Dict):
        """Monitor job progress"""
        max_wait_time = 1800  # 30 minutes max
        check_interval = 10   # Check every 10 seconds
        start_time = time.time()
        
        logger.info(f"   ⏳ Monitoring job {job_id}")
        
        while time.time() - start_time < max_wait_time:
            try:
                # Check job status
                status_response = requests.get(
                    f"{self.orchestrator_url}/api/job/{job_id}",
                    timeout=5
                )
                
                if status_response.status_code == 200:
                    job_data = status_response.json()
                    status = job_data["job"]["status"]
                    
                    if status in ["completed", "failed", "error", "timeout"]:
                        logger.info(f"   ✅ Job {job_id} finished with status: {status}")
                        
                        # Run autonomous data collection on success
                        if status == "completed":
                            self._run_autonomous_collection()
                        
                        return
                    
                    # Still running, show progress
                    elapsed = int(time.time() - start_time)
                    logger.debug(f"   Job {job_id} still running... ({elapsed}s elapsed)")
                
            except Exception as e:
                logger.debug(f"   Status check failed: {e}")
            
            time.sleep(check_interval)
        
        logger.warning(f"   ⚠️ Job {job_id} timed out after {max_wait_time} seconds")
    
    def _run_autonomous_collection(self):
        """Trigger LLM's autonomous data collection"""
        try:
            logger.info("   🤖 Triggering autonomous data collection...")
            
            response = requests.post(
                "http://localhost:5000/agent/collect_data",
                json={"strategy": "auto"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   ✅ Autonomous collection: {data.get('samples_collected', 0)} samples")
            else:
                logger.warning("   ⚠️ Autonomous collection failed")
                
        except Exception as e:
            logger.debug(f"   Autonomous collection error: {e}")
    
    def stop(self):
        """Stop automated training"""
        self.running = False
        logger.info("🛑 Auto-trainer stopped")
    
    def get_status(self) -> Dict:
        """Get current status"""
        return {
            "running": self.running,
            "current_job": self.current_job,
            "history_count": len(self.training_history),
            "next_training": str(schedule.next_run()) if schedule.jobs else None,
            "topics_available": len(self.training_topics)
        }
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get training history"""
        return self.training_history[-limit:] if self.training_history else []
    
    def add_topic(self, topic: str):
        """Add a new training topic"""
        if topic not in self.training_topics:
            self.training_topics.append(topic)
            logger.info(f"➕ Added topic: {topic}")
    
    def remove_topic(self, topic: str):
        """Remove a training topic"""
        if topic in self.training_topics:
            self.training_topics.remove(topic)
            logger.info(f"➖ Removed topic: {topic}")

# Web interface for monitoring
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
trainer = AutoTrainer()

@app.route('/')
def dashboard():
    """Web dashboard for auto-trainer"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 Auto Trainer Dashboard</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                margin: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            header {
                text-align: center;
                margin-bottom: 40px;
            }
            h1 {
                color: #333;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .status-card {
                background: #f8f9fa;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 30px;
                border-left: 5px solid #007bff;
            }
            .status-running { border-left-color: #28a745; background: #d4edda; }
            .status-stopped { border-left-color: #dc3545; background: #f8d7da; }
            .btn {
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                cursor: pointer;
                margin: 5px;
                font-weight: 600;
                transition: all 0.2s;
            }
            .btn-start { background: #28a745; color: white; }
            .btn-start:hover { background: #218838; }
            .btn-stop { background: #dc3545; color: white; }
            .btn-stop:hover { background: #c82333; }
            .btn-topic { background: #17a2b8; color: white; }
            .history-item {
                padding: 15px;
                border-bottom: 1px solid #dee2e6;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .history-item:hover { background: #f8f9fa; }
            .topic-badge {
                display: inline-block;
                padding: 4px 12px;
                background: #e9ecef;
                border-radius: 20px;
                font-size: 14px;
                margin: 2px;
            }
            .controls {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                margin: 20px 0;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border: 1px solid #dee2e6;
            }
            .stat-value {
                font-size: 2.5em;
                font-weight: bold;
                color: #007bff;
                margin: 10px 0;
            }
            .log {
                background: #1a202c;
                color: #e2e8f0;
                padding: 15px;
                border-radius: 6px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 14px;
                max-height: 300px;
                overflow-y: auto;
                white-space: pre-wrap;
            }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🤖 Auto Trainer Dashboard</h1>
                <p>Automated LLM Training Pipeline</p>
            </header>
            
            <div id="statusCard" class="status-card">
                <h3>System Status</h3>
                <p>Loading...</p>
            </div>
            
            <div class="controls">
                <button class="btn btn-start" onclick="startTraining()">▶️ Start Auto-Training</button>
                <button class="btn btn-stop" onclick="stopTraining()">⏹️ Stop Auto-Training</button>
                <button class="btn btn-topic" onclick="addRandomTopic()">➕ Add Random Topic</button>
                <button class="btn" onclick="runNow()">⚡ Train Now</button>
                <button class="btn" onclick="triggerAutonomous()">🤖 Autonomous Collection</button>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-label">Training Cycles</div>
                    <div class="stat-value" id="cycleCount">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Topics</div>
                    <div class="stat-value" id="topicCount">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Next Training</div>
                    <div class="stat-value" id="nextTraining">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Current Job</div>
                    <div class="stat-value" id="currentJob">None</div>
                </div>
            </div>
            
            <div class="controls">
                <input type="number" id="interval" value="30" min="5" max="1440" style="padding: 10px;">
                <button class="btn" onclick="updateInterval()">🔄 Set Interval (minutes)</button>
                <input type="text" id="newTopic" placeholder="Add custom topic" style="padding: 10px; flex-grow: 1;">
                <button class="btn" onclick="addCustomTopic()">📚 Add Topic</button>
            </div>
            
            <h3>📋 Training History</h3>
            <div id="history"></div>
            
            <h3>📝 Available Topics</h3>
            <div id="topics"></div>
            
            <h3>📊 Live Log</h3>
            <div class="log" id="logOutput">
                Starting...
            </div>
        </div>
        
        <script>
            let autoRefresh = true;
            
            async function updateStatus() {
                try {
                    const response = await axios.get('/api/status');
                    const data = response.data;
                    
                    // Update status card
                    const statusCard = document.getElementById('statusCard');
                    statusCard.className = `status-card ${data.running ? 'status-running' : 'status-stopped'}`;
                    statusCard.innerHTML = `
                        <h3>${data.running ? '🟢 AUTO-TRAINING RUNNING' : '🔴 AUTO-TRAINING STOPPED'}</h3>
                        <p>Current job: ${data.current_job || 'None'}</p>
                        <p>Next training: ${data.next_training || 'Not scheduled'}</p>
                        <p>Topics available: ${data.topics_available}</p>
                        <p>Training cycles completed: ${data.history_count}</p>
                    `;
                    
                    // Update stats
                    document.getElementById('cycleCount').textContent = data.history_count;
                    document.getElementById('topicCount').textContent = data.topics_available;
                    document.getElementById('nextTraining').textContent = data.next_training ? 'Soon' : '-';
                    document.getElementById('currentJob').textContent = data.current_job ? data.current_job.slice(0, 8) + '...' : 'None';
                    
                } catch (error) {
                    console.error('Status update failed:', error);
                }
            }
            
            async function updateHistory() {
                try {
                    const response = await axios.get('/api/history?limit=10');
                    const history = response.data;
                    
                    let html = '';
                    history.forEach(item => {
                        html += `
                            <div class="history-item">
                                <div>
                                    <strong>${new Date(item.timestamp).toLocaleString()}</strong><br>
                                    <span style="color: #666;">${item.topic}</span>
                                </div>
                                <div>
                                    <span style="background: #${item.status === 'completed' ? 'd4edda' : 'f8d7da'}; 
                                          padding: 4px 12px; border-radius: 4px;">
                                        ${item.status}
                                    </span>
                                </div>
                            </div>
                        `;
                    });
                    
                    document.getElementById('history').innerHTML = html || '<p>No training history yet.</p>';
                    
                } catch (error) {
                    console.error('History update failed:', error);
                }
            }
            
            async function updateTopics() {
                try {
                    const response = await axios.get('/api/topics');
                    const topics = response.data.topics;
                    
                    let html = '';
                    topics.forEach(topic => {
                        html += `<span class="topic-badge">${topic}</span>`;
                    });
                    
                    document.getElementById('topics').innerHTML = html;
                    
                } catch (error) {
                    console.error('Topics update failed:', error);
                }
            }
            
            async function startTraining() {
                const interval = document.getElementById('interval').value;
                try {
                    await axios.post('/api/start', { interval_minutes: parseInt(interval), immediate: true });
                    alert('✅ Auto-training started!');
                    updateStatus();
                } catch (error) {
                    alert('❌ Failed to start: ' + error.response?.data?.error || error.message);
                }
            }
            
            async function stopTraining() {
                try {
                    await axios.post('/api/stop');
                    alert('🛑 Auto-training stopped!');
                    updateStatus();
                } catch (error) {
                    alert('❌ Failed to stop: ' + error.response?.data?.error || error.message);
                }
            }
            
            async function runNow() {
                try {
                    await axios.post('/api/train_now');
                    alert('⚡ Training started now!');
                    updateStatus();
                } catch (error) {
                    alert('❌ Failed to start training: ' + error.response?.data?.error || error.message);
                }
            }
            
            async function triggerAutonomous() {
                try {
                    await axios.post('/api/autonomous');
                    alert('🤖 Autonomous collection triggered!');
                } catch (error) {
                    alert('❌ Failed: ' + error.response?.data?.error || error.message);
                }
            }
            
            async function addRandomTopic() {
                const randomTopics = [
                    'astronomy', 'genetics', 'cryptography', 'blockchain', 
                    'metaphysics', 'ethics', 'linguistics', 'anthropology'
                ];
                const topic = randomTopics[Math.floor(Math.random() * randomTopics.length)];
                await addTopic(topic);
            }
            
            async function addCustomTopic() {
                const topic = document.getElementById('newTopic').value.trim();
                if (topic) {
                    await addTopic(topic);
                    document.getElementById('newTopic').value = '';
                }
            }
            
            async function addTopic(topic) {
                try {
                    await axios.post('/api/add_topic', { topic: topic });
                    alert(`➕ Added topic: ${topic}`);
                    updateTopics();
                } catch (error) {
                    alert('❌ Failed to add topic');
                }
            }
            
            async function updateInterval() {
                const interval = document.getElementById('interval').value;
                try {
                    await axios.post('/api/set_interval', { interval_minutes: parseInt(interval) });
                    alert(`🔄 Interval updated to ${interval} minutes`);
                    updateStatus();
                } catch (error) {
                    alert('❌ Failed to update interval');
                }
            }
            
            // Live log
            let logEntries = [];
            async function updateLog() {
                try {
                    const response = await axios.get('/api/log');
                    const logs = response.data.logs;
                    
                    if (logs.length > logEntries.length) {
                        const newLogs = logs.slice(logEntries.length);
                        newLogs.forEach(log => {
                            logEntries.push(`[${new Date(log.timestamp).toLocaleTimeString()}] ${log.message}`);
                        });
                        
                        if (logEntries.length > 50) {
                            logEntries = logEntries.slice(-50);
                        }
                        
                        document.getElementById('logOutput').textContent = logEntries.join('\\n');
                        document.getElementById('logOutput').scrollTop = document.getElementById('logOutput').scrollHeight;
                    }
                } catch (error) {
                    console.error('Log update failed:', error);
                }
            }
            
            // Auto-refresh
            function refreshAll() {
                if (autoRefresh) {
                    updateStatus();
                    updateHistory();
                    updateTopics();
                    updateLog();
                }
            }
            
            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                refreshAll();
                setInterval(refreshAll, 2000); // Update every 2 seconds
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get auto-trainer status"""
    return jsonify(trainer.get_status())

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get training history"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify(trainer.get_history(limit))

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Get available topics"""
    return jsonify({"topics": trainer.training_topics})

@app.route('/api/start', methods=['POST'])
def start_training():
    """Start auto-training"""
    data = request.get_json() or {}
    interval = data.get('interval_minutes', 30)
    immediate = data.get('immediate', True)
    
    success = trainer.start_auto_training(interval_minutes=interval, immediate=immediate)
    
    if success:
        return jsonify({"status": "started", "interval": interval})
    else:
        return jsonify({"error": "Failed to start training"}), 500

@app.route('/api/stop', methods=['POST'])
def stop_training():
    """Stop auto-training"""
    trainer.stop()
    return jsonify({"status": "stopped"})

@app.route('/api/train_now', methods=['POST'])
def train_now():
    """Run training cycle immediately"""
    if trainer.running:
        threading.Thread(target=trainer.run_training_cycle, daemon=True).start()
        return jsonify({"status": "started"})
    else:
        return jsonify({"error": "Auto-trainer not running"}), 400

@app.route('/api/autonomous', methods=['POST'])
def trigger_autonomous():
    """Trigger autonomous collection"""
    try:
        trainer._run_autonomous_collection()
        return jsonify({"status": "triggered"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add_topic', methods=['POST'])
def add_topic():
    """Add a new topic"""
    data = requests.get_json() or {}
    topic = data.get('topic')
    
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
    
    trainer.add_topic(topic)
    return jsonify({"status": "added", "topic": topic})

@app.route('/api/set_interval', methods=['POST'])
def set_interval():
    """Update training interval"""
    # Note: Currently requires restart to change interval
    return jsonify({"status": "Interval changes require restart"})

@app.route('/api/log', methods=['GET'])
def get_log():
    """Get recent logs"""
    # This is a simplified log - in production, use proper logging
    logs = [
        {"timestamp": datetime.now().isoformat(), "message": "System ready"}
    ]
    return jsonify({"logs": logs})

def main():
    """Start the auto-trainer with web interface"""
    print("=" * 60)
    print("🤖 AUTO TRAINER STARTING")
    print("=" * 60)
    print()
    print("📡 Orchestrator: http://localhost:8080")
    print("🤖 Auto Trainer: http://localhost:8081")
    print()
    print("⚡ Training will start automatically")
    print("📊 Dashboard available at http://localhost:8081")
    print()
    print("=" * 60)
    
    # Start auto-training immediately
    trainer.start_auto_training(interval_minutes=30, immediate=True)
    
    # Start web interface
    app.run(host='0.0.0.0', port=8081, debug=True, threaded=True)

if __name__ == "__main__":
    main()
