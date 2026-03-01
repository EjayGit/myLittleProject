import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

// Constants - ONLY KEEPING USED ENDPOINTS
const API_BASE = 'http://localhost:5000';
const API_ENDPOINTS = {
	HEALTH: '/health',
	STATUS: '/status',
	INIT: '/init',
	TRAIN: '/train',
	STOP_TRAINING: '/stop_training',
	INFER: '/infer',
	SET_TRAIN_DATA: '/set_train_data',
	AGENT_DATA_STATUS: '/agent/data_status',
	AGENT_COLLECT_DATA: '/agent/collect_data',
	AGENT_SELF_IMPROVE: '/agent/self_improve'
	// REMOVED: AGENT_DEBUG, AGENT_QUICK_SETUP, AGENT_ADD_TRAINING_DATA
};

const DEFAULT_TRAINING_PARAMS = {
	epochs: 15,
	batch_size: 32,
	lr: 0.001
};

const MIN_TRAINING_TOKENS = 65;

const setError = msg => window.alert(msg);

// Custom hook for API calls
const useApi = () => {
	const [loading, setLoading] = useState({});
	const [error, setError] = useState(null);

	const callApi = useCallback(async (endpoint, options = {}, loadingKey = null) => {
		if (loadingKey) {
			setLoading(prev => ({ ...prev, [loadingKey]: true }));
		}
		setError(null);

		try {
			const response = await fetch(endpoint, {
				...options,
				headers: {
					'Content-Type': 'application/json',
					...options.headers,
				},
			});

			if (!response.ok) {
				const errorData = await response.json().catch(() => ({}));
				throw new Error(errorData.message || `HTTP ${response.status}`);
			}

			return await response.json();
		} catch (err) {
			setError(err.message);
			throw err;
		} finally {
			if (loadingKey) {
				setLoading(prev => ({ ...prev, [loadingKey]: false }));
			}
		}
	}, []);

	const clearError = useCallback(() => setError(null), []);

	return { callApi, loading, error, clearError };
};

// Sub-components for better organization
const ConnectionStatus = ({ connectionStatus, onRetry }) => {
	const getStatusConfig = () => {
		switch (connectionStatus) {
			case 'connected': return { color: '🟢', text: 'Connected to Backend', className: 'connected' };
			case 'testing': return { color: '🟡', text: 'Testing Connection...', className: 'testing' };
			case 'disconnected': return { color: '🔴', text: 'Disconnected from Backend', className: 'disconnected' };
			default: return { color: '⚪', text: 'Unknown Status', className: '' };
		}
	};

	const { color, text, className } = getStatusConfig();

	return (
		<div className={`connection-status ${className}`}>
			<span className="status-indicator">{color} {text}</span>
			{connectionStatus === 'disconnected' && (
				<button onClick={onRetry} className="retry-button">Retry</button>
			)}
		</div>
	);
};

const ErrorMessage = ({ error, onDismiss }) => {
	if (!error) return null;

	return (
		<div className="error-message">
			<span className="error-text">{error}</span>
			<button onClick={onDismiss} className="dismiss-error">×</button>
		</div>
	);
};

const DataCollectionPanel = ({
	onCollectData,
	onSelfImprove,
	loading,
	isInitialized,
	dataStatus
}) => {
	return (
		<div className="panel agentic-panel">
			<h2>🧠 Autonomous AI Control</h2>
			<div className="agentic-controls">
				<div className="data-controls">
					<h3>Data Collection</h3>
					<button
						onClick={() => onCollectData('web')}
						disabled={loading.dataCollection || !isInitialized}
					>
						{loading.dataCollection ? 'Collecting...' : '🌐 Web Data'}
					</button>
					<button
						onClick={() => onCollectData('wikipedia')}
						disabled={loading.dataCollection || !isInitialized}
					>
						{loading.dataCollection ? 'Collecting...' : '📚 Wikipedia'}
					</button>
					<button
						onClick={() => onCollectData('knowledge')}
						disabled={loading.dataCollection || !isInitialized}
					>
						{loading.dataCollection ? 'Collecting...' : '🧠 Knowledge Base'}
					</button>
					<button
						onClick={() => onCollectData('synthetic')}
						disabled={loading.dataCollection || !isInitialized}
					>
						{loading.dataCollection ? 'Collecting...' : '🔧 Synthetic Data'}
					</button>
					<button
						onClick={() => onCollectData('auto')}
						disabled={loading.dataCollection || !isInitialized}
					>
						{loading.dataCollection ? 'Collecting...' : '🤖 Auto Collect'}
					</button>
				</div>

				<div className="self-improvement">
					<h3>Self-Improvement</h3>
					<button
						onClick={onSelfImprove}
						disabled={!isInitialized || loading.selfImprove}
						className="improve-button"
					>
						{loading.selfImprove ? 'Improving...' : '🎯 Trigger Self-Improvement'}
					</button>
					<p className="help-text">
						{dataStatus.autonomousImprovementReady
							? 'AI is ready for autonomous improvement'
							: 'AI will self-improve when needed'}
					</p>
				</div>
			</div>

			<div className="data-status">
				<h3>📊 Current Status</h3>
				<div className="status-grid">
					<div className="status-item">
						<span className="label">Training Data:</span>
						<span className="value">{dataStatus.trainingDataSize} tokens</span>
					</div>
					<div className="status-item">
						<span className="label">Vocabulary:</span>
						<span className="value">{dataStatus.vocabularySize} words</span>
					</div>
					<div className="status-item">
						<span className="label">Unique Words:</span>
						<span className="value">{dataStatus.uniqueWords || 0}</span>
					</div>
					<div className="status-item">
						<span className="label">Ready for Training:</span>
						<span className="value">
							{dataStatus.trainingDataSize >= MIN_TRAINING_TOKENS ? '✅' : '❌'}
						</span>
					</div>
					<div className="status-item">
						<span className="label">Learning Goals:</span>
						<span className="value">{dataStatus.currentLearningGoals || 0}</span>
					</div>
					<div className="status-item">
						<span className="label">Training Cycles:</span>
						<span className="value">{dataStatus.trainingCyclesCompleted || 0}</span>
					</div>
				</div>
			</div>
		</div>
	);
};

const LossChart = ({ lossHistory }) => {
	if (lossHistory.length === 0) return null;

	const maxLoss = Math.max(...lossHistory);
	const minLoss = Math.min(...lossHistory);
	const range = maxLoss - minLoss || 1;

	return (
		<div className="loss-chart">
			<h3>Training Loss</h3>
			<div className="chart-container">
				{lossHistory.map((loss, index) => (
					<div
						key={index}
						className="chart-bar"
						style={{
							height: `${((loss - minLoss) / range) * 100}%`,
							width: `${100 / lossHistory.length}%`
						}}
						title={`Step ${index + 1}: ${loss.toFixed(4)}`}
					></div>
				))}
			</div>
			<div className="chart-labels">
				<span>Max: {maxLoss.toFixed(4)}</span>
				<span>Min: {minLoss.toFixed(4)}</span>
				<span>Current: {lossHistory[lossHistory.length - 1].toFixed(4)}</span>
			</div>
		</div>
	);
};

// Main App Component
function App() {
	// State management
	const [inputText, setInputText] = useState('');
	const [outputText, setOutputText] = useState('');
	const [isTraining, setIsTraining] = useState(false);
	const [isInitialized, setIsInitialized] = useState(false);
	const [trainingProgress, setTrainingProgress] = useState(0);
	const [trainingText, setTrainingText] = useState('');
	const [trainingLoss, setTrainingLoss] = useState(null);
	const [lossHistory, setLossHistory] = useState([]);
	const [trainingParams, setTrainingParams] = useState(DEFAULT_TRAINING_PARAMS);
	const [vocabSize, setVocabSize] = useState(0);
	const [connectionStatus, setConnectionStatus] = useState('testing');
	const [dataStatus, setDataStatus] = useState({
		trainingDataSize: 0,
		vocabularySize: 0,
		uniqueWords: 0,
		strategies: [],
		autonomousImprovementReady: false,
		trainingCyclesCompleted: 0,
		currentLearningGoals: 0
	});
	const [startTime, setStartTime] = useState(null);

	// Custom hooks
	const { callApi, loading, error, clearError } = useApi();

	// Enhanced connection testing
	const testConnection = useCallback(async () => {
		try {
			setConnectionStatus('testing');

			const controller = new AbortController();
			const timeoutId = setTimeout(() => controller.abort(), 5000);

			await callApi(`${API_BASE}${API_ENDPOINTS.HEALTH}`, { signal: controller.signal });

			clearTimeout(timeoutId);
			setConnectionStatus('connected');
			return true;
		} catch (err) {
			setConnectionStatus('disconnected');
			console.error('Connection test failed:', err);
			return false;
		}
	}, [callApi]);

	// API functions - ONLY USED ONES
	const fetchStatus = useCallback(async () => {
		try {
			const data = await callApi(`${API_BASE}${API_ENDPOINTS.STATUS}`);
			setIsInitialized(data.initialized);
			setIsTraining(data.training_in_progress);
			setTrainingLoss(data.training_loss);
			setTrainingProgress(data.training_progress || 0);
			setVocabSize(data.vocab_size || 0);
		} catch (err) {
			console.error('Status fetch error:', err);
		}
	}, [callApi]);

	const fetchDataStatus = useCallback(async () => {
		try {
			const data = await callApi(`${API_BASE}${API_ENDPOINTS.AGENT_DATA_STATUS}`);
			setDataStatus({
				trainingDataSize: data.training_data_size || 0,
				vocabularySize: data.vocabulary_size || 0,
				uniqueWords: data.unique_words || 0,
				strategies: data.data_collection_strategies || [],
				autonomousImprovementReady: data.autonomous_improvement_ready || false,
				trainingCyclesCompleted: data.training_cycles_completed || 0,
				currentLearningGoals: data.current_learning_goals || 0
			});
		} catch (err) {
			console.error('Data status fetch error:', err);
		}
	}, [callApi]);

	// Effects
	useEffect(() => {
		testConnection();
	}, [testConnection]);

	useEffect(() => {
		if (connectionStatus !== 'connected') return;

		const updateStatus = () => {
			fetchStatus();
			fetchDataStatus();
		};

		updateStatus();
		const interval = setInterval(updateStatus, 5000);

		return () => clearInterval(interval);
	}, [connectionStatus, fetchStatus, fetchDataStatus]);

	useEffect(() => {
		if (!isTraining) return;

		const interval = setInterval(() => {
			if (connectionStatus === 'connected') {
				fetchStatus();

				if (trainingLoss !== null) {
					setLossHistory(prev => [...prev, trainingLoss].slice(-10));
				}
			}
		}, 1000);

		return () => clearInterval(interval);
	}, [isTraining, connectionStatus, trainingLoss, fetchStatus]);

	// Event handlers - ONLY USED ONES
	const collectData = async (strategy = 'auto') => {
		try {
			const data = await callApi(
				`${API_BASE}${API_ENDPOINTS.AGENT_COLLECT_DATA}`,
				{
					method: 'POST',
					body: JSON.stringify({ strategy })
				},
				'dataCollection'
			);

			alert(`✅ Collected ${data.samples_collected} samples\n📝 Added ${data.new_words_added} new words\n🔢 Added ${data.tokens_added} tokens\n📚 Vocabulary now: ${data.vocab_size} words`);

			fetchStatus();
			fetchDataStatus();
		} catch (err) {
			console.error('Data collection error:', err);
		}
	};

	const initializeModel = async () => {
		try {
			const data = await callApi(
				`${API_BASE}${API_ENDPOINTS.INIT}`,
				{ method: 'POST' },
				'initialization'
			);

			setIsInitialized(data.initialized);
			setVocabSize(data.vocab_size || 0);
			alert('🎯 Model initialized successfully! Agentic AI is ready.');
			fetchDataStatus();
		} catch (err) {
			console.error('Initialization error:', err);
		}
	};

	const triggerSelfImprovement = async () => {
		try {
			await callApi(
				`${API_BASE}${API_ENDPOINTS.AGENT_SELF_IMPROVE}`,
				{ method: 'POST' },
				'selfImprove'
			);

			alert('🤖 Autonomous improvement cycle started! The AI is collecting data and optimizing itself.');
			fetchDataStatus();
		} catch (err) {
			console.error('Self-improvement error:', err);
		}
	};

	const handleInference = async () => {
		try {
			const data = await callApi(
				`${API_BASE}${API_ENDPOINTS.INFER}`,
				{
					method: 'POST',
					body: JSON.stringify({
						text: inputText,
						max_tokens: 20,
						temperature: 0.7,
						top_p: 0.9,
					})
				},
				'inference'
			);
			setOutputText(data.output);
		} catch (err) {
			console.error('Inference error:', err);
		}
	};

	const setTrainingData = async () => {
		try {
			const data = await callApi(
				`${API_BASE}${API_ENDPOINTS.SET_TRAIN_DATA}`,
				{
					method: 'POST',
					body: JSON.stringify({ text: trainingText })
				},
				'trainingData'
			);

			alert(`📚 Training data set! ${data.tokens} tokens, ${data.new_words_added || 0} new words added`);
			setVocabSize(data.vocab_size || vocabSize);
			fetchDataStatus();
		} catch (err) {
			console.error('Set training data error:', err);
		}
	};

	const startTraining = async () => {
		if (connectionStatus !== 'connected') {
			setError('No connection to backend');
			return;
		}

		if (dataStatus.trainingDataSize < MIN_TRAINING_TOKENS) {
			setError(`Not enough training data. Need at least ${MIN_TRAINING_TOKENS} tokens, but only have ${dataStatus.trainingDataSize}. Collect more data first.`);
			return;
		}

		if (!isInitialized) {
			setError('Model must be initialized before training. Please initialize the model first.');
			return;
		}

		if (vocabSize < 10) {
			setError('Vocabulary too small. Collect data first.');
			return;
		}

		try {
			setIsTraining(true);
			setStartTime(Date.now());
			setLossHistory([]);
			clearError();

			await callApi(
				`${API_BASE}${API_ENDPOINTS.TRAIN}`,
				{
					method: 'POST',
					body: JSON.stringify(trainingParams)
				}
			);

			setTimeout(() => {
				alert('Training initialised successfully!');
			}, 1000);

		} catch (err) {
			setIsTraining(false);
			console.error('Training error:', err);
		}
	};

	const stopTraining = async () => {
		try {
			await callApi(`${API_BASE}${API_ENDPOINTS.STOP_TRAINING}`, { method: 'POST' });
			setIsTraining(false);
		} catch (err) {
			console.error('Error stopping training:', err);
		}
	};

	const handleParamChange = (param, value) => {
		setTrainingParams(prev => ({
			...prev,
			[param]: value
		}));
	};

	const handleKeyPress = (e) => {
		if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
			handleInference();
		}
	};

	const estimatedTime = startTime && trainingProgress > 0
		? Math.round(((Date.now() - startTime) / trainingProgress) * (100 - trainingProgress) / 1000)
		: null;

	return (
		<div className="App">
			<header>
				<h1>🤖 Agentic Mamba AI</h1>
				<ConnectionStatus
					connectionStatus={connectionStatus}
					onRetry={testConnection}
				/>
			</header>

			<div className="container">
				<ErrorMessage error={error} onDismiss={clearError} />

				{connectionStatus !== 'connected' && (
					<div className="warning-message">
						<strong>Backend Connection Required</strong>
						<p>Make sure your Flask server is running on port 5000.</p>
						<button onClick={() => window.location.reload()}>Retry Connection</button>
					</div>
				)}

				<DataCollectionPanel
					onCollectData={collectData}
					onSelfImprove={triggerSelfImprovement}
					loading={loading}
					isInitialized={isInitialized}
					dataStatus={dataStatus}
				/>

				<div className="panel">
					<h2>📚 Training</h2>

					<div className="model-initialization">
						<h3>Model Initialization</h3>
						<button
							onClick={initializeModel}
							disabled={isInitialized || loading.initialization || connectionStatus !== 'connected'}
							className="init-button"
						>
							{loading.initialization ? 'Initializing...' :
								isInitialized ? '✅ Model Initialized' : '🚀 Initialize Model'}
						</button>
						<p className="help-text">
							{isInitialized
								? 'Model is ready for training and inference'
								: 'Initialize the model before starting training'}
						</p>
					</div>

					<textarea
						value={trainingText}
						onChange={(e) => setTrainingText(e.target.value)}
						placeholder="Enter training text here..."
						rows={5}
					/>
					<button
						onClick={setTrainingData}
						disabled={isTraining || loading.trainingData || !isInitialized || connectionStatus !== 'connected'}
						className="data-button"
					>
						{loading.trainingData ? 'Setting Data...' : '💾 Set Training Data'}
					</button>

					<div className="training-params">
						<h3>Training Parameters</h3>
						<div className="param-group">
							<label>
								Epochs (1-100):
								<input
									type="number"
									min="1"
									max="100"
									value={trainingParams.epochs}
									onChange={(e) => handleParamChange('epochs', parseInt(e.target.value) || 4)}
									disabled={isTraining}
								/>
							</label>

							<label>
								Batch Size (1-256):
								<input
									type="number"
									min="1"
									max="256"
									value={trainingParams.batch_size}
									onChange={(e) => handleParamChange('batch_size', parseInt(e.target.value) || 32)}
									disabled={isTraining}
								/>
							</label>

							<label>
								Learning Rate:
								<input
									type="number"
									step="0.0001"
									min="0.0001"
									max="0.1"
									value={trainingParams.lr}
									onChange={(e) => handleParamChange('lr', parseFloat(e.target.value) || 0.001)}
									disabled={isTraining}
								/>
							</label>
						</div>
					</div>

					<div className="training-controls">
						<button
							onClick={startTraining}
							disabled={isTraining || !isInitialized || connectionStatus !== 'connected' || dataStatus.trainingDataSize < MIN_TRAINING_TOKENS}
							className={`train-button ${isTraining ? 'training-active' : ''}`}
						>
							{isTraining ? '🔄 Training...' : '🎯 Start Training'}
						</button>

						{isTraining && (
							<button onClick={stopTraining} className="stop-button">
								⏹️ Stop Training
							</button>
						)}
					</div>

					{isTraining && (
						<div className="training-progress">
							<h4>Training Progress</h4>
							<p>Progress: {trainingProgress}%</p>
							<progress value={trainingProgress} max="100"></progress>

							{estimatedTime && (
								<p>Estimated time remaining: {Math.floor((estimatedTime/60))} minutes</p>
							)}

							{trainingLoss !== null && (
								<p>Current loss: {trainingLoss.toFixed(4)}</p>
							)}

							<LossChart lossHistory={lossHistory} />
						</div>
					)}
				</div>

				<div className="panel">
					<h2>🔮 Inference</h2>
					<textarea
						value={inputText}
						onChange={(e) => setInputText(e.target.value)}
						onKeyDown={handleKeyPress}
						placeholder="Enter text to generate completion... Try: 'the cat', 'a bird', 'the sun' (Ctrl+Enter to generate)"
						rows={3}
					/>
					<button
						onClick={handleInference}
						disabled={!isInitialized || isTraining || loading.inference || connectionStatus !== 'connected'}
						className="inference-button"
					>
						{loading.inference ? '✨ Generating...' : '🚀 Run Inference'}
					</button>

					<div className="output">
						<h3>Generated Text:</h3>
						<div className="output-text">
							{outputText || 'No output yet. Initialize the model, add training data, train, then try inference!'}
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}

export default App;
