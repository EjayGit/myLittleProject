from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import os
import json
import logging
from collections import defaultdict, deque
import numpy as np
import re
import requests
from datetime import datetime, timedelta
import time
from model import SimplifiedMamba
from bs4 import BeautifulSoup
import random
import wikipedia
# from transformers import AutoTokenizer

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_FILE = 'mamba_model.pt'
VOCAB_FILE = 'vocab.json'
VOCAB_SIZE = 50000
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
CONTEXT_SIZE = 64
TRAIN_DATA_FILE = 'train_data.json'

# Global state with thread safety
model = None
optimizer = None
criterion = nn.CrossEntropyLoss()
training_in_progress = False
training_progress = 0
training_loss = None
vocab = defaultdict(lambda: len(vocab))
reverse_vocab = {}
vocab_lock = threading.Lock()
train_data = []
initialized = False
training_thread = None
training_lock = threading.Lock()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

# Agentic AI State
class AgenticState:
    def __init__(self):
        self.learning_goals = deque(maxlen=10)
        self.performance_history = deque(maxlen=100)
        self.self_evaluation_scores = {}
        self.autonomous_tasks = []
        self.adaptation_strategies = []
        self.knowledge_base = {}
        self.last_self_improvement = datetime.now()
        self.training_cycles_completed = 0
        
    def add_learning_goal(self, goal, priority=1):
        self.learning_goals.append({
            'goal': goal,
            'priority': priority,
            'created': datetime.now(),
            'completed': False
        })
        
    def evaluate_performance(self, loss, accuracy_metrics=None):
        performance_score = 1.0 / (loss + 1e-8)
        self.performance_history.append({
            'timestamp': datetime.now(),
            'loss': loss,
            'score': performance_score,
            'metrics': accuracy_metrics or {}
        })
        return performance_score
        
    def should_self_improve(self):
        if len(self.performance_history) < 5:
            return False
        recent_scores = [p['score'] for p in list(self.performance_history)[-5:]]
        avg_score = np.mean(recent_scores)
        time_since_improvement = (datetime.now() - self.last_self_improvement).total_seconds()
        return avg_score < 0.5 or time_since_improvement > 3600

agentic_state = AgenticState()

def save_train_data():
    """Save training data to file"""
    try:
        with open(TRAIN_DATA_FILE, 'w') as f:
            json.dump(train_data, f)
        logger.info(f"💾 Training data saved: {len(train_data)} tokens")
    except Exception as e:
        logger.error(f"Failed to save training data: {str(e)}")

def load_train_data():
    """Load training data from file"""
    global train_data
    try:
        if os.path.exists(TRAIN_DATA_FILE):
            with open(TRAIN_DATA_FILE, 'r') as f:
                loaded_data = json.load(f)
                train_data.clear()
                train_data.extend(loaded_data)
            logger.info(f"📂 Training data loaded: {len(train_data)} tokens")
            return True
        else:
            logger.info("No saved training data found")
            return False
    except Exception as e:
        logger.error(f"Failed to load training data: {str(e)}")
        return False

def init_vocab():
    global vocab, reverse_vocab
    with vocab_lock:
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        reverse_vocab = {v: k for k, v in vocab.items()}

def init_model():
    global model, optimizer, initialized
    if not initialized:
        model = SimplifiedMamba(
            vocab_size=VOCAB_SIZE,
            d_model=EMBEDDING_DIM,
            n_layer=NUM_LAYERS
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        initialized = True
        agentic_state.add_learning_goal("Improve text coherence and fluency", priority=2)
        agentic_state.add_learning_goal("Expand vocabulary coverage", priority=1)
        agentic_state.add_learning_goal("Optimize inference speed", priority=3)
        logger.info("Model initialized on device: %s", device)

def save_artifacts():
    with vocab_lock:
        try:
            agentic_state_save = {
                'learning_goals': list(agentic_state.learning_goals),
                'performance_history': list(agentic_state.performance_history),
                'self_evaluation_scores': agentic_state.self_evaluation_scores,
                'autonomous_tasks': agentic_state.autonomous_tasks,
                'adaptation_strategies': agentic_state.adaptation_strategies,
                'knowledge_base': agentic_state.knowledge_base,
                'last_self_improvement': agentic_state.last_self_improvement.isoformat(),
                'training_cycles_completed': agentic_state.training_cycles_completed
            }
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab_size': VOCAB_SIZE,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'context_size': CONTEXT_SIZE,
                'agentic_state': agentic_state_save
            }, MODEL_FILE)
            
            with open(VOCAB_FILE, 'w') as f:
                json.dump(dict(vocab), f)
            
            logger.info("Artifacts saved successfully")
        except Exception as e:
            logger.error("Failed to save artifacts: %s", str(e))

def load_artifacts():
    global model, optimizer, vocab, reverse_vocab, initialized, agentic_state
    
    if os.path.exists(VOCAB_FILE):
        try:
            with open(VOCAB_FILE, 'r') as f:
                loaded_vocab = json.load(f)
                with vocab_lock:
                    vocab.clear()
                    vocab.update(loaded_vocab)
                    reverse_vocab = {v: k for k, v in loaded_vocab.items()}
            logger.info("Vocabulary loaded with %d tokens", len(vocab))
        except Exception as e:
            logger.error("Failed to load vocabulary: %s", str(e))
    
    init_model()
    
    if os.path.exists(MODEL_FILE):
        try:
            import collections
            from torch.serialization import add_safe_globals
            add_safe_globals([collections.deque])
            checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)
            
            config_mismatch = []
            for param in ['vocab_size', 'embedding_dim', 'hidden_dim', 'num_layers', 'context_size']:
                if param in checkpoint and checkpoint[param] != globals()[param.upper()]:
                    config_mismatch.append(f"{param}: saved={checkpoint[param]}, current={globals()[param.upper()]}")
            
            if config_mismatch:
                logger.warning("Model configuration mismatch: %s", ", ".join(config_mismatch))
                return
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'agentic_state' in checkpoint:
                try:
                    agentic_data = checkpoint['agentic_state']
                    if 'learning_goals' in agentic_data:
                        agentic_state.learning_goals = deque(agentic_data['learning_goals'], maxlen=10)
                    if 'performance_history' in agentic_data:
                        agentic_state.performance_history = deque(agentic_data['performance_history'], maxlen=100)
                    
                    simple_attrs = ['self_evaluation_scores', 'autonomous_tasks', 
                                  'adaptation_strategies', 'knowledge_base', 'training_cycles_completed']
                    for attr in simple_attrs:
                        if attr in agentic_data:
                            setattr(agentic_state, attr, agentic_data[attr])
                    
                    if 'last_self_improvement' in agentic_data:
                        last_imp = agentic_data['last_self_improvement']
                        if isinstance(last_imp, str):
                            agentic_state.last_self_improvement = datetime.fromisoformat(last_imp)
                        else:
                            agentic_state.last_self_improvement = last_imp
                    
                    logger.info("Agentic state loaded successfully")
                except Exception as e:
                    logger.error("Failed to load agentic state: %s", str(e))
                    agentic_state = AgenticState()
            
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error("Failed to load model: %s", str(e))
            init_model()

def autonomous_improvement_cycle():
    """Agent autonomously improves itself based on self-evaluation"""
    if not agentic_state.should_self_improve():
        return
    
    logger.info("Starting autonomous improvement cycle")
    
    try:
        # Determine data collection strategy based on current needs
        data_strategies = adaptive_data_strategy()
        
        # Collect new data based on strategy
        new_data = []
        if "expand_vocabulary" in data_strategies:
            new_data.extend(collect_from_knowledge_base())
            new_data.extend(generate_synthetic_data())
        
        if "diversify_content" in data_strategies:
            new_data.extend(collect_from_wikipedia())
            new_data.extend(collect_from_web())
        
        if "reinforce_basics" in data_strategies:
            # Focus on fundamental concepts
            basic_data = [
                "Basic concepts form the foundation of advanced understanding.",
                "Learning fundamentals enables mastery of complex topics.",
                "Strong basics support efficient problem solving approaches.",
                "Fundamental principles guide application in various contexts.",
                "Core concepts provide the building blocks for specialization."
            ]
            new_data.extend(basic_data)
        
        if "advanced_topics" in data_strategies or not data_strategies:
            # Default to comprehensive collection
            new_data.extend(autonomous_data_collection())
        
        # Process and incorporate new data - PROPERLY update vocabulary
        tokens_added = 0
        new_words_added = 0
        
        if new_data:
            with vocab_lock:
                for text in new_data:
                    words = text.split()
                    for word in words:
                        # Add new words to vocabulary if we have space
                        word = word.lower()
                        if word not in vocab and len(vocab) < VOCAB_SIZE:
                            vocab[word] = len(vocab)
                            new_words_added += 1
                        
                        # Add token to training data
                        if word in vocab:
                            train_data.append(vocab[word])
                            tokens_added += 1
                
                # Update reverse vocabulary
                reverse_vocab.clear()
                reverse_vocab.update({v: k for k, v in vocab.items()})
            
            logger.info("Autonomously added %d new words and %d tokens to training data", 
                       new_words_added, tokens_added)
                
            # Add learning goal about data expansion
            agentic_state.add_learning_goal(
                f"Integrate {len(new_data)} new data samples into knowledge base", 
                priority=2
            )
        
        # Adjust learning strategy based on performance
        if len(agentic_state.performance_history) > 10:
            recent_performance = [p['score'] for p in list(agentic_state.performance_history)[-10:]]
            avg_performance = np.mean(recent_performance)
            
            if avg_performance < 0.3:
                agentic_state.add_learning_goal("Address performance degradation with enhanced training strategy", priority=1)
            elif avg_performance > 0.7:
                agentic_state.add_learning_goal("Explore advanced learning techniques and complex data", priority=3)
        
        agentic_state.last_self_improvement = datetime.now()
        agentic_state.training_cycles_completed += 1
        
        logger.info("Autonomous improvement cycle completed successfully")
        
    except Exception as e:
        logger.error("Autonomous improvement cycle failed: %s", str(e))
        
def autonomous_data_collection():
    """Agent autonomously collects training data from multiple sources"""
    try:
        collected_data = []
        
        # Strategy 1: Web scraping (common knowledge)
        web_data = collect_from_web()
        if web_data:
            collected_data.extend(web_data)
            logger.info(f"Collected {len(web_data)} samples from web")
        
        # Strategy 2: Wikipedia articles
        wiki_data = collect_from_wikipedia()
        if wiki_data:
            collected_data.extend(wiki_data)
            logger.info(f"Collected {len(wiki_data)} samples from Wikipedia")
        
        # Strategy 3: Predefined knowledge base
        kb_data = collect_from_knowledge_base()
        if kb_data:
            collected_data.extend(kb_data)
            logger.info(f"Collected {len(kb_data)} samples from knowledge base")
        
        # Strategy 4: Generated examples based on current vocabulary
        generated_data = generate_synthetic_data()
        if generated_data:
            collected_data.extend(generated_data)
            logger.info(f"Generated {len(generated_data)} synthetic samples")
        
        logger.info(f"Autonomous data collection completed: {len(collected_data)} total samples")
        return collected_data
        
    except Exception as e:
        logger.error(f"Autonomous data collection failed: {str(e)}")
        return []

def collect_from_web():
    """Collect data from public domain texts and common knowledge sources"""
    try:
        # Public domain texts and common knowledge sentences
        web_sentences = [
            "The sun rises in the east and sets in the west.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Plants convert carbon dioxide into oxygen through photosynthesis.",
            "The Earth revolves around the sun in approximately 365 days.",
            "Gravity is the force that attracts objects toward each other.",
            "Electricity flows through conductors like copper and aluminum.",
            "The human body is composed of approximately 60% water.",
            "Computers process information using binary code consisting of ones and zeros.",
            "Languages evolve over time through cultural interactions and technological changes.",
            "Machine learning algorithms improve their performance through training on data.",
            "Neural networks are inspired by the structure of the human brain.",
            "Python is a popular programming language for artificial intelligence development.",
            "Data science involves extracting insights from large datasets.",
            "Natural language processing enables computers to understand human language.",
            "The internet connects computers worldwide through a network of networks.",
            "Cloud computing allows access to computing resources over the internet.",
            "Blockchain technology provides secure and transparent transaction records.",
            "Renewable energy sources include solar, wind, and hydroelectric power.",
            "Climate change refers to long-term shifts in global weather patterns.",
            "Biodiversity is essential for maintaining healthy ecosystems on Earth."
        ]
        
        # Add some variation
        variations = []
        for sentence in web_sentences:
            words = sentence.split()
            if len(words) > 5:
                # Create variations by taking different parts of sentences
                variations.append(' '.join(words[:len(words)//2]))
                variations.append(' '.join(words[len(words)//2:]))
        
        return web_sentences + variations[:10]  # Limit variations
        
    except Exception as e:
        logger.error(f"Web data collection failed: {str(e)}")
        return []

def collect_from_wikipedia():
    """Collect data from Wikipedia on relevant topics"""
    try:
        wikipedia.set_rate_limiting(rate_limit=True, min_wait=timedelta(seconds=1.5))
        topics = [
            "Subatomic particle",
            "Atom",
            "Quantum mechanics",
            "Electromagnetic field",
            "Engineering",
            "Chemistry",
            "Biology", 
            "Physics",
            "Mathematics",
            "Science",
            "Quantum computing",
            "Programming",
            "Algorithm",
            "Neural network",
            "Data science",
            "Deep learning",
            "Python programming language",
            "Computer science",
            "Natural language processing",
            "Machine learning", 
            "Artificial intelligence",
            "Linear combination",
            "Bell's theorem",
            "Wave function collapse",
            "Correlation",
            "Spin (physics)",
            "Position (geometry)",
            "Quantum entanglement",
            "Processor (computing)", 
            "Central processing unit",
            "Machine code",
            "Assembly language",
            "Translator (computing)",
            "Interpreter (computing)",
            "Source code",
            "Computer",
            "Hardware",
            "Compiler",
            "Execution (computing)",
            "Momentum",
            "Software",
            "Computer program",
            "Polarization (waves)",
            "Strong interaction",
            "Weak interaction",
            "Radioactive decay",
            "Antimatter",
            "Antiparticle",
            "Antiproton",
            "Ionization",
            "Salt (chemistry)",
            "Electronegativity",
            "Coulomb's law",
            "Ionic bonding",
            "Covalent bond",
            "Crystal",
            "Chemical bond",
            "Force",
            "Molecule",
            "Ion",
            "Plasma (physics)",
            "Matter",
            "Fermion",
            "Boson",
            "Hadron",
            "Electric charge",
            "Classical electromagnetism",
            "Quantum electrodynamics",
            "Electromagnetic radiation",
            "Energy",
            "Stress (mechanics)",
            "General relativity",
            "Time",
            "Spacetime",
            "Gravity",
            "Mass",
            "Neutron",
            "Proton",
            "Quark",
            "Electron",
            "Positron",
            "Elementary particle",
            "Neutrino",
            "Neutrino oscillation",
            "Particle decay",
            "Special relativity",
            "Quantum superposition"
        ]
        
        wiki_data = []
        for topic in topics:  # Consider limit of 3 topics to avoid rate limiting
            try:
                # Get summary
                summary = wikipedia.summary(topic, sentences=0)
                sentences = [s.strip() for s in summary.split('. ') if len(s) > 20]
                wiki_data.extend(sentences)
                
                # Add some related context
                related_terms = [
                    f"{topic} is a field of study",
                    f"Researchers work on {topic}",
                    f"Applications of {topic} include various domains",
                    f"The development of {topic} has progressed significantly"
                ]
                wiki_data.extend(related_terms)
                
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation by taking the first option
                try:
                    summary = wikipedia.summary(e.options[0], sentences=2)
                    sentences = [s.strip() for s in summary.split('. ') if len(s) > 20]
                    wiki_data.extend(sentences)
                except:
                    continue
            except Exception as e:
                logger.warning(f"Could not fetch Wikipedia data for {topic}: {str(e)}")
                continue
        
        return wiki_data
        
    except Exception as e:
        logger.error(f"Wikipedia data collection failed: {str(e)}")
        return []

def collect_from_knowledge_base():
    """Collect from predefined knowledge base focused on AI and technology"""
    try:
        knowledge_base = [
            # AI and ML concepts
            "Machine learning models learn patterns from data without explicit programming.",
            "Supervised learning uses labeled datasets to train algorithms.",
            "Unsupervised learning finds hidden patterns in unlabeled data.",
            "Reinforcement learning trains agents through rewards and punishments.",
            "Deep learning uses neural networks with multiple layers.",
            "Computer vision enables machines to interpret visual information.",
            "Robotics combines hardware and software to create autonomous machines.",
            
            # Programming concepts
            "Functions encapsulate reusable pieces of code with specific purposes.",
            "Object-oriented programming organizes code into objects and classes.",
            "Data structures like arrays and dictionaries store information efficiently.",
            "Algorithms are step-by-step procedures for solving problems.",
            "Debugging involves finding and fixing errors in computer programs.",
            "Version control systems track changes to source code over time.",
            
            # Technology trends
            "Cloud computing provides scalable resources over the internet.",
            "Internet of Things connects physical devices to the digital world.",
            "Cybersecurity protects computer systems from malicious attacks.",
            "Big data refers to extremely large datasets that require special processing.",
            "Quantum computing uses quantum mechanics to perform complex calculations.",
            
            # General knowledge
            "Scientific method involves observation, hypothesis, and experimentation.",
            "Critical thinking evaluates information objectively and logically.",
            "Problem solving requires identifying issues and implementing solutions.",
            "Communication skills are essential for effective collaboration.",
            "Continuous learning helps adapt to technological advancements."
        ]
        
        return knowledge_base
        
    except Exception as e:
        logger.error(f"Knowledge base collection failed: {str(e)}")
        return []

def ensure_basic_vocabulary():
    """Ensure basic vocabulary words are included"""
    basic_words = [
        'the', 'cat', 'sits', 'on', 'mat', 'dog', 'runs', 'in', 'park', 
        'a', 'bird', 'flies', 'sky', 'sun', 'is', 'bright', 'and', 'warm',
        'people', 'work', 'at', 'their', 'jobs', 'children', 'play', 'with', 'toys',
        'water', 'important', 'for', 'life', 'food', 'tastes', 'good', 'when', 'hungry',
        'moon', 'appears', 'night', 'cars', 'drive', 'road', 'trees', 'grow', 'forest',
        'flowers', 'bloom', 'spring', 'rain', 'falls', 'from', 'clouds', 'wind', 'blows', 'through',
        'students', 'learn', 'school', 'teachers', 'help', 'books', 'contain', 'stories', 'knowledge',
        'music', 'makes', 'happy', 'houses', 'provide', 'shelter', 'families', 'computer', 'processes', 
        'information', 'quickly', 'house', 'red', 'blue', 'green', 'yellow', 'big', 'small', 'fast', 'slow',
        'hot', 'cold', 'day', 'time', 'year', 'man', 'woman', 'child', 'city', 'country', 'world'
    ]
    
    words_added = 0
    with vocab_lock:
        for word in basic_words:
            if word not in vocab and len(vocab) < VOCAB_SIZE:
                vocab[word] = len(vocab)
                words_added += 1
        
        # Update reverse vocabulary
        reverse_vocab.clear()
        reverse_vocab.update({v: k for k, v in vocab.items()})
    
    logger.info(f"Ensured basic vocabulary: {words_added} new words added")
    return words_added

def generate_synthetic_data():
    """Generate synthetic data based on current vocabulary and patterns"""
    try:
        if len(vocab) < 10:  # Need some vocabulary to work with
            return []
            
        synthetic_data = []
        common_patterns = [
            "The {noun} is {adjective}",
            "We can use {noun} for {purpose}",
            "{Subject} {verb} {object}",
            "Understanding {concept} helps with {application}",
            "The development of {technology} has transformed {field}"
        ]
        
        # Get some words from current vocabulary for context
        with vocab_lock:
            known_words = [word for word in list(vocab.keys())[:50] if len(word) > 3 and word not in ['<PAD>', '<UNK>']]
        
        if len(known_words) < 10:
            return []
        
        # Generate some synthetic sentences
        for pattern in common_patterns:
            for _ in range(3):  # Generate 3 variations per pattern
                filled_pattern = pattern
                if '{noun}' in filled_pattern and known_words:
                    filled_pattern = filled_pattern.replace('{noun}', random.choice(known_words), 1)
                if '{adjective}' in filled_pattern:
                    filled_pattern = filled_pattern.replace('{adjective}', random.choice(['important', 'useful', 'essential', 'valuable', 'critical']))
                if '{purpose}' in filled_pattern:
                    filled_pattern = filled_pattern.replace('{purpose}', random.choice(['learning', 'development', 'analysis', 'research', 'innovation']))
                if '{Subject}' in filled_pattern:
                    filled_pattern = filled_pattern.replace('{Subject}', random.choice(['Researchers', 'Scientists', 'Engineers', 'Developers', 'Students']))
                if '{verb}' in filled_pattern:
                    filled_pattern = filled_pattern.replace('{verb}', random.choice(['study', 'develop', 'create', 'analyze', 'build']))
                if '{object}' in filled_pattern and known_words:
                    filled_pattern = filled_pattern.replace('{object}', random.choice(known_words))
                if '{concept}' in filled_pattern and known_words:
                    filled_pattern = filled_pattern.replace('{concept}', random.choice(known_words))
                if '{application}' in filled_pattern:
                    filled_pattern = filled_pattern.replace('{application}', random.choice(['problem solving', 'data analysis', 'system design', 'research']))
                if '{technology}' in filled_pattern and known_words:
                    filled_pattern = filled_pattern.replace('{technology}', random.choice(known_words))
                if '{field}' in filled_pattern:
                    filled_pattern = filled_pattern.replace('{field}', random.choice(['technology', 'science', 'engineering', 'research']))
                
                synthetic_data.append(filled_pattern + ".")
        
        return synthetic_data[:15]  # Limit output
        
    except Exception as e:
        logger.error(f"Synthetic data generation failed: {str(e)}")
        return []

def adaptive_data_strategy():
    """Determine what type of data to collect based on current needs"""
    try:
        strategies = []
        
        # Analyze current vocabulary size
        vocab_size = len(vocab)
        if vocab_size < 100:
            strategies.append("expand_vocabulary")
        if vocab_size < 500:
            strategies.append("diversify_content")
        
        # Analyze performance to determine needs
        if agentic_state.performance_history:
            recent_losses = [p['loss'] for p in list(agentic_state.performance_history)[-3:]]
            avg_recent_loss = np.mean(recent_losses) if recent_losses else 10.0
            
            if avg_recent_loss > 5.0:
                strategies.append("reinforce_basics")
            elif avg_recent_loss < 2.0:
                strategies.append("advanced_topics")
        
        logger.info(f"Adaptive data strategy: {strategies}")
        return strategies
        
    except Exception as e:
        logger.error(f"Adaptive data strategy failed: {str(e)}")
        return ["general_knowledge"]

@app.route('/agent/collect_data', methods=['POST'])
def trigger_data_collection():
    """Manually trigger autonomous data collection"""
    try:
        data = request.get_json() or {}
        strategy = data.get('strategy', 'auto')
        
        if strategy == 'web':
            collected = collect_from_web()
        elif strategy == 'wikipedia':
            collected = collect_from_wikipedia()
        elif strategy == 'knowledge':
            collected = collect_from_knowledge_base()
        elif strategy == 'synthetic':
            collected = generate_synthetic_data()
        else:
            collected = autonomous_data_collection()
        
        # Add to training data AND update vocabulary
        tokens_added = 0
        new_words_added = 0
        
        if collected:
            with vocab_lock:
                for text in collected:
                    words = text.split()
                    for word in words:
                        # Add new words to vocabulary if we have space
                        word = word.lower()
                        if word not in vocab and len(vocab) < VOCAB_SIZE:
                            vocab[word] = len(vocab)
                            new_words_added += 1
                        
                        # Add token to training data
                        if word in vocab:
                            train_data.append(vocab[word])
                            tokens_added += 1
                
                # Update reverse vocabulary
                reverse_vocab.clear()
                reverse_vocab.update({v: k for k, v in vocab.items()})
        
        logger.info(f"Data collection: {len(collected)} samples, {new_words_added} new words, {tokens_added} tokens")
        
        return jsonify({
            'status': 'success',
            'samples_collected': len(collected),
            'tokens_added': tokens_added,
            'new_words_added': new_words_added,
            'vocab_size': len(vocab),
            'strategy_used': strategy
        })
        
    except Exception as e:
        logger.error(f"Manual data collection failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Data collection failed: {str(e)}'
        }), 500

@app.route('/agent/data_status', methods=['GET'])
def data_status():
    """Get status of training data and collection capabilities"""
    return jsonify({
        'training_data_size': len(train_data),
        'vocabulary_size': len(vocab),
        'unique_words': len([word for word in vocab.keys() if word not in ['<PAD>', '<UNK>']]),
        'data_collection_strategies': [
            'web', 'wikipedia', 'knowledge', 'synthetic', 'auto'
        ],
        'autonomous_improvement_ready': agentic_state.should_self_improve(),
        'last_self_improvement': agentic_state.last_self_improvement.isoformat(),
        'training_cycles_completed': agentic_state.training_cycles_completed,
        'current_learning_goals': len(agentic_state.learning_goals)
    })

@app.route('/agent/self_improve', methods=['POST'])
def trigger_self_improvement():
    """Manually trigger autonomous self-improvement cycle"""
    try:
        if training_in_progress:
            return jsonify({
                'status': 'error',
                'message': 'Cannot start self-improvement during training'
            }), 409
        
        # Start autonomous improvement in a separate thread
        improvement_thread = threading.Thread(
            target=autonomous_improvement_cycle,
            daemon=True
        )
        improvement_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Autonomous improvement cycle started',
            'training_cycles': agentic_state.training_cycles_completed + 1,
            'should_improve': agentic_state.should_self_improve()
        })
        
    except Exception as e:
        logger.error(f"Self-improvement trigger failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Self-improvement failed: {str(e)}'
        }), 500
    
@app.route('/stop_training', methods=['POST'])
def stop_training():
    """Stop the current training process"""
    global training_in_progress
    try:
        with training_lock:
            training_in_progress = False
        logger.info("Training stop requested")
        return jsonify({
            'status': 'success',
            'message': 'Training stop requested'
        })
    except Exception as e:
        logger.error(f"Failed to stop training: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to stop training: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Flask server is running',
        'timestamp': datetime.now().isoformat(),
        'initialized': initialized,
        'training_in_progress': training_in_progress
    })


# Your existing endpoints continue here...
@app.route('/status')
def status():
    return jsonify({
        'initialized': initialized,
        'training_in_progress': training_in_progress,
        'training_progress': training_progress,
        'training_loss': training_loss,
        'vocab_size': len(vocab),
        'device': str(device),
        'agentic_mode': True,
        'learning_goals_count': len(agentic_state.learning_goals)
    })

@app.route('/debug_state', methods=['GET'])
def debug_state():
    return jsonify({
        'vocab_size': len(vocab),
        'vocab_first_10': dict(list(vocab.items())[:10]) if vocab else {},
        'train_data_size': len(train_data),
        'train_data_first_10': train_data[:10] if train_data else [],
        'initialized': initialized,
        'model_loaded': model is not None,
        'vocab_file_exists': os.path.exists(VOCAB_FILE),
        'train_data_file_exists': os.path.exists(TRAIN_DATA_FILE),
        'model_file_exists': os.path.exists(MODEL_FILE)
    })

@app.route('/init', methods=['POST'])
def initialize():
    if not initialized:
        try:
            init_vocab()
            init_model()
            load_artifacts()
            return jsonify({
                'status': 'success',
                'initialized': initialized,
                'vocab_size': len(vocab),
                'device': str(device),
                'agentic_goals': len(agentic_state.learning_goals)
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Initialization failed: {str(e)}'
            }), 500
    return jsonify({
        'status': 'success',
        'initialized': initialized,
        'vocab_size': len(vocab),
        'device': str(device),
        'agentic_goals': len(agentic_state.learning_goals)
    })

@app.route('/set_train_data', methods=['POST'])
def set_train_data():
    global train_data
    
    if training_in_progress:
        return jsonify({
            'status': 'error',
            'message': 'Cannot set training data during training'
        }), 400
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'status': 'error', 'message': 'No text provided'}), 400
    
    text = data.get('text', '')
    words = text.split()[:50000]
    
    with vocab_lock:
        tokens = []
        new_words = 0
        for word in words:
            word = word.lower()
            if word not in vocab:
                if len(vocab) >= VOCAB_SIZE:
                    tokens.append(vocab['<UNK>'])
                    continue
                vocab[word] = len(vocab)
                new_words += 1
            tokens.append(vocab[word])
        
        reverse_vocab.clear()
        reverse_vocab.update({v: k for k, v in vocab.items()})
        train_data = tokens
    
    logger.info("Training data set: %d tokens, %d new words added", len(train_data), new_words)
    return jsonify({
        'status': 'success',
        'tokens': len(train_data),
        'vocab_size': len(vocab),
        'new_words_added': new_words
    })

@app.route('/train', methods=['POST'])
def train():
    global training_thread
    
    if training_in_progress:
        return jsonify({
            'status': 'error',
            'message': 'Training already in progress'
        }), 409
    
    data = request.get_json() or {}
    epochs = data.get('epochs', 3)
    batch_size = data.get('batch_size', 32)
    lr = data.get('lr', 0.001)
    
    if not isinstance(epochs, int) or epochs <= 0 or epochs > 100:
        return jsonify({
            'status': 'error',
            'message': 'Invalid epochs value (must be 1-100)'
        }), 400
    
    try:
        # Validate we can actually train before starting thread
        if not train_data or len(train_data) < CONTEXT_SIZE + 1:
            return jsonify({
                'status': 'error', 
                'message': f'Not enough training data. Need {CONTEXT_SIZE + 1} tokens, have {len(train_data) if train_data else 0}'
            }), 400
            
        if not initialized:
            return jsonify({
                'status': 'error',
                'message': 'Model not initialized'
            }), 400
        
        logger.info(f"Starting training thread with epochs={epochs}, batch_size={batch_size}, lr={lr}")
        
        # Start training in a thread but handle exceptions properly
        training_thread = threading.Thread(
            target=run_training_with_error_handling,
            kwargs={
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
            },
            daemon=True
        )
        training_thread.start()
        
        logger.info("Training thread started successfully")
        
        return jsonify({
            'status': 'success',
            'message': 'Training started',
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'data_size': len(train_data)
        })
        
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to start training: {str(e)}'
        }), 500

def train_model(epochs=4, batch_size=32, lr=0.001, autonomous=False):
    global training_in_progress, training_progress, training_loss
    
    try:
        logger.info(f"=== TRAINING STARTING ===")
        logger.info(f"Training data size: {len(train_data)} tokens")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
        logger.info(f"Model initialized: {initialized}")
        logger.info(f"Vocabulary size: {len(vocab)}")
        logger.info(f"Device: {device}")
        
        # Enhanced validation
        if not initialized:
            error_msg = "Model not initialized"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not train_data:
            error_msg = "No training data available"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if len(train_data) < CONTEXT_SIZE + 1:
            error_msg = f"Insufficient training data. Need at least {CONTEXT_SIZE + 1} tokens, have {len(train_data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if len(vocab) < 10:
            error_msg = f"Vocabulary too small: {len(vocab)} words. Need at least 10 words."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Test model forward pass before training
        logger.info("Testing model forward pass...")
        try:
            test_input = torch.tensor([[1, 2, 3]], dtype=torch.long).to(device)
            with torch.no_grad():
                test_output = model(test_input)
            logger.info(f"Model test successful. Output shape: {test_output.shape}")
        except Exception as e:
            error_msg = f"Model forward pass failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Prepare actual training data with context windows
        sequences = []
        targets = []
        for i in range(len(train_data) - CONTEXT_SIZE):
            sequences.append(train_data[i:i+CONTEXT_SIZE])
            targets.append(train_data[i+CONTEXT_SIZE])
        
        logger.info(f"Created {len(sequences)} training sequences")
        
        if not sequences:
            raise ValueError("Not enough data to create training sequences")
        
        # Convert to tensors
        inputs = torch.tensor(sequences, dtype=torch.long).to(device)
        targets_tensor = torch.tensor(targets, dtype=torch.long).to(device)
        
        logger.info(f"Input tensor shape: {inputs.shape}")
        logger.info(f"Target tensor shape: {targets_tensor.shape}")
        
        dataset = torch.utils.data.TensorDataset(inputs, targets_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=min(batch_size, len(dataset)),
            shuffle=True
        )
        
        logger.info(f"Dataset prepared: {len(dataset)} samples, {len(dataloader)} batches")
        
        # Set training state
        with training_lock:
            training_in_progress = True
            training_progress = 0
            training_loss = None
        
        total_batches = len(dataloader)
        logger.info(f"Starting training loop with {epochs} epochs, {total_batches} batches per epoch")
        
        # TRAINING LOOP
        for epoch in range(epochs):
            epoch_loss = 0.0
            model.train()
            logger.info(f"=== Epoch {epoch+1}/{epochs} ===")
            
            for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
                if not training_in_progress:
                    logger.info("Training stopped by user")
                    break
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_inputs)
                
                # outputs shape: [batch_size, seq_len, vocab_size]
                # We only care about predicting the next token from the last position
                last_outputs = outputs[:, -1, :]  # [batch_size, vocab_size]

                # batch_targets shape: [batch_size]
                loss = criterion(last_outputs, batch_targets)

                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                epoch_loss += loss.item()

                # Update progress
                with training_lock:
                    batch_progress = ((epoch * total_batches) + batch_idx + 1) / (epochs * total_batches) * 100
                    training_progress = int(batch_progress)
                    training_loss = epoch_loss / (batch_idx + 1)
                
                # Log progress
                if batch_idx % max(1, total_batches // 5) == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}, Progress: {training_progress}%")
            
            if not training_in_progress:
                break
                
            avg_epoch_loss = epoch_loss / total_batches
            with training_lock:
                training_loss = avg_epoch_loss
            
            logger.info(f"Epoch {epoch+1}/{epochs} completed | Loss: {avg_epoch_loss:.4f}")
        
        logger.info("🎉 Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        return False
    finally:
        with training_lock:
            training_in_progress = False
        try:
            save_artifacts()
            logger.info("💾 Artifacts saved after training")
        except Exception as e:
            logger.error("💥 Failed to save artifacts after training: %s", str(e))


def run_training_with_error_handling(epochs, batch_size, lr):
    """Wrapper function to handle training errors properly"""
    try:
        success = train_model(epochs=epochs, batch_size=batch_size, lr=lr, autonomous=False)
        if success:
            logger.info("Training completed successfully")
        else:
            logger.error("Training failed")
    except Exception as e:
        logger.error(f"Training thread crashed: {str(e)}", exc_info=True)
    finally:
        # Ensure training state is reset
        with training_lock:
            training_in_progress = False

@app.route('/infer', methods=['POST'])
def infer():
    try:
        if not initialized:
            return jsonify({
                'status': 'error',
                'message': 'Model not initialized. Please click "Initialize Model" first.'
            }), 400
            
        if training_in_progress:
            return jsonify({
                'status': 'error',
                'message': 'Inference unavailable during training'
            }), 423
            
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No text provided'
            }), 400
        
        text = data.get('text', '').strip()
        max_tokens = min(data.get('max_tokens', 15), 50)  # Reduced max tokens
        temperature = max(0.3, min(data.get('temperature', 0.7), 1.5))  # Adjusted temperature range
        
        if not text:
            return jsonify({
                'status': 'error',
                'message': 'Input text is empty'
            }), 400
        
        logger.info(f"Inference request: '{text}' (max_tokens: {max_tokens}, temp: {temperature})")
        
        # Check if we have enough vocabulary and training
        if len(vocab) <= 50: # or len(train_data) < 100:
            return jsonify({
                'status': 'success',
                'input': text,
                'output': "Model needs more training. Please collect more data and train for longer.",
                'tokens_generated': 0,
                'note': 'insufficient_training'
            })
        
        # Tokenize input text
        words = text.split()
        with vocab_lock:
            input_ids = []
            for word in words:
                word = word.lower()
                if word in vocab:
                    input_ids.append(vocab[word])
                else:
                    input_ids.append(vocab['<UNK>'])
        
        # Ensure we have enough context
        if len(input_ids) < CONTEXT_SIZE:
            # Pad at the beginning to maintain context
            padding = [vocab['<PAD>']] * (CONTEXT_SIZE - len(input_ids))
            input_ids = padding + input_ids
        else:
            # Take only the last CONTEXT_SIZE tokens
            input_ids = input_ids[-CONTEXT_SIZE:]
        
        # Convert to tensor and move to device
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # Generate text using the model with improved sampling
        model.eval()
        generated_tokens = []
        repeated_count = 0
        last_token = None
        
        with torch.no_grad():
            current_sequence = input_tensor
            
            for i in range(max_tokens):
                # Get model predictions
                outputs = model(current_sequence)
                next_token_logits = outputs[0, -1, :] / temperature
                
                # IMPROVED: Add repetition penalty to prevent "The The The" pattern
                repetition_penalty = 1.2
                for token_id in set(generated_tokens[-4:]):  # Penalize recent tokens
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty
                
                # IMPROVED: Avoid unknown and padding tokens when possible
                if len(generated_tokens) > 0:  # Only after first token
                    next_token_logits[vocab['<UNK>']] = -float('inf')
                    next_token_logits[vocab['<PAD>']] = -float('inf')
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_token_id = torch.multinomial(probabilities, num_samples=1).item()
                
                # Track repetitions
                if next_token_id == last_token:
                    repeated_count += 1
                else:
                    repeated_count = 0
                last_token = next_token_id
                
                # Stop if we're repeating too much
                if repeated_count > 3:  # Reduced from 5 to 3
                    break
                
                generated_tokens.append(next_token_id)
                
                # Update the sequence for next prediction
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
                current_sequence = torch.cat([current_sequence, next_token_tensor], dim=1)
                
                # Keep only the last CONTEXT_SIZE tokens
                if current_sequence.shape[1] > CONTEXT_SIZE:
                    current_sequence = current_sequence[:, -CONTEXT_SIZE:]
        
        # Convert generated tokens back to text
        with vocab_lock:
            generated_words = []
            for token_id in generated_tokens:
                if token_id in reverse_vocab:
                    word = reverse_vocab[token_id]
                    # Skip padding tokens in output
                    if word != '<PAD>':
                        generated_words.append(word)
                else:
                    generated_words.append('<UNK>')
        
        output_text = ' '.join(generated_words)
        
        # IMPROVED: Basic post-processing to handle empty or repetitive outputs
        if not output_text.strip() or output_text.strip() == text.strip():
            output_text = "Model needs more training to generate diverse outputs."
        elif len(set(generated_words)) < 3 and len(generated_words) > 5:  # Too repetitive
            output_text = "Model is repeating too much. Try more training with diverse data."
        
        logger.info(f"Inference completed: generated {len(generated_words)} words: '{output_text}'")
        
        return jsonify({
            'status': 'success',
            'input': text,
            'output': output_text,
            'tokens_generated': len(generated_words),
            'temperature_used': temperature
        })
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Inference failed: {str(e)}'
        }), 500


@app.route('/data_debug', methods=['GET'])
def data_debug():
    """Debug data persistence"""
    return jsonify({
        'train_data_file_exists': os.path.exists('train_data.json'),
        'train_data_size': len(train_data),
        'train_data_sample': train_data[:10] if train_data else [],
        'vocab_size': len(vocab),
        'vocab_file_exists': os.path.exists(VOCAB_FILE),
        'model_file_exists': os.path.exists(MODEL_FILE),
        'files_in_directory': os.listdir('.') if os.path.exists('.') else []
    })

@app.route('/server_info', methods=['GET'])
def server_info():
    """Get server process information"""
    import os
    return jsonify({
        'process_id': os.getpid(),
        'training_data_size': len(train_data),
        'train_data_memory_id': id(train_data),
        'train_data_file_exists': os.path.exists('train_data.json'),
        'current_working_directory': os.getcwd()
    })

if __name__ == '__main__':
    try:
        init_vocab()
        load_artifacts()
        load_train_data()
        
        logger.info("=== 🚀 Agentic AI Server Starting ===")
        logger.info(f"Device: {device}")
        logger.info(f"Training data: {len(train_data)} tokens")
        logger.info(f"Vocabulary: {len(vocab)} words")
        logger.info(f"Model initialized: {initialized}")
        
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
        
    except Exception as e:
        logger.exception("💥 Server failed to start: %s", str(e))
  
