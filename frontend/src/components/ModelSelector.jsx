import { useState, useEffect } from 'react';
import { getDatasetPreview, trainModel, getModels } from '../services/api';
import { Play, Activity, Settings2, Box, RefreshCw, Sliders } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ResultsDashboard from './ResultsDashboard';

const MODELS = [
  { id: 'RandomForest', name: 'Random Forest', type: 'Ensemble' },
  { id: 'SVM', name: 'Support Vector Machine/SVR', type: 'Linear/Non-Linear' },
  { id: 'KNN', name: 'K-Nearest Neighbors', type: 'Distance' },
  { id: 'Linear/Logistic', name: 'Linear / Logistic Regression', type: 'Linear' },
];

export default function ModelSelector() {
  const [columns, setColumns] = useState([]);
  const [target, setTarget] = useState('');
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);
  const [training, setTraining] = useState(false);
  const [experiments, setExperiments] = useState([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState(null);
  
  // Hyperparameters
  const [hyperparams, setHyperparams] = useState({});

  useEffect(() => {
    getDatasetPreview().then(res => {
      setColumns(res.data.columns);
      if (res.data.columns && res.data.columns.length > 0) {
        setTarget(res.data.columns[res.data.columns.length - 1]);
      }
    });
    fetchExperiments();
  }, []);

  // Reset hyperparams when model changes
  useEffect(() => {
    if (selectedModel === 'RandomForest') setHyperparams({ n_estimators: 100 });
    else if (selectedModel === 'SVM') setHyperparams({ C: 1.0, kernel: 'rbf' });
    else if (selectedModel === 'KNN') setHyperparams({ n_neighbors: 5 });
    else setHyperparams({});
  }, [selectedModel]);

  const fetchExperiments = () => {
    getModels().then(res => setExperiments(res.data));
  };

  const handleTrain = async () => {
    setTraining(true);
    try {
      await trainModel({
        model_name: selectedModel,
        target_column: target,
        hyperparameters: hyperparams
      });
      fetchExperiments();
    } catch (err) {
      console.error(err);
      alert("Training failed to start: " + (err.response?.data?.detail || err.message));
    } finally {
      setTraining(false);
    }
  };

  if (selectedExperimentId) {
    return (
      <ResultsDashboard 
        experimentId={selectedExperimentId} 
        onBack={() => {
          setSelectedExperimentId(null);
          fetchExperiments();
        }} 
      />
    );
  }

  return (
    <div className="space-y-8">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="modern-card p-8 relative overflow-hidden"
      >
        <div className="absolute -left-20 -top-20 w-64 h-64 bg-brand-50 rounded-full blur-3xl pointer-events-none"></div>
        <h2 className="text-2xl font-bold text-text-900 mb-6 flex items-center">
          <Settings2 className="mr-3 text-brand-500" />
          Train New Model
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8 relative z-10">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-text-600 mb-2">Target Variable</label>
              <select 
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                className="w-full bg-surface-50 border border-surface-300 rounded-xl px-4 py-3 text-text-900 shadow-sm focus:border-brand-500 focus:ring-1 focus:ring-brand-500 focus:outline-none transition-shadow"
              >
                {columns.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            
            {Object.keys(hyperparams).length > 0 && (
              <div className="bg-surface-50 p-5 rounded-xl border border-surface-200 shadow-sm">
                <h3 className="text-sm font-bold text-text-800 mb-4 flex items-center">
                   <Sliders size={16} className="mr-2 text-brand-500" /> Hyperparameters
                </h3>
                <div className="space-y-4">
                  {selectedModel === 'RandomForest' && (
                    <div>
                      <label className="block text-xs font-semibold text-text-500 mb-1">N Estimators (Trees)</label>
                      <input type="number" min="10" max="1000" value={hyperparams.n_estimators || 100} onChange={e => setHyperparams({...hyperparams, n_estimators: parseInt(e.target.value)})} className="w-full border border-surface-300 rounded-lg px-3 py-2 text-sm focus:border-brand-500 outline-none" />
                    </div>
                  )}
                  {selectedModel === 'SVM' && (
                    <>
                      <div>
                        <label className="block text-xs font-semibold text-text-500 mb-1">C (Regularization)</label>
                        <input type="number" step="0.1" value={hyperparams.C || 1.0} onChange={e => setHyperparams({...hyperparams, C: parseFloat(e.target.value)})} className="w-full border border-surface-300 rounded-lg px-3 py-2 text-sm focus:border-brand-500 outline-none" />
                      </div>
                      <div>
                        <label className="block text-xs font-semibold text-text-500 mb-1">Kernel</label>
                        <select value={hyperparams.kernel || 'rbf'} onChange={e => setHyperparams({...hyperparams, kernel: e.target.value})} className="w-full border border-surface-300 rounded-lg px-3 py-2 text-sm focus:border-brand-500 outline-none bg-white">
                          <option value="rbf">RBF</option>
                          <option value="linear">Linear</option>
                          <option value="poly">Polynomial</option>
                        </select>
                      </div>
                    </>
                  )}
                  {selectedModel === 'KNN' && (
                    <div>
                      <label className="block text-xs font-semibold text-text-500 mb-1">N Neighbors</label>
                      <input type="number" min="1" max="50" value={hyperparams.n_neighbors || 5} onChange={e => setHyperparams({...hyperparams, n_neighbors: parseInt(e.target.value)})} className="w-full border border-surface-300 rounded-lg px-3 py-2 text-sm focus:border-brand-500 outline-none" />
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
          
          <div>
            <label className="block text-sm font-medium text-text-600 mb-2">Algorithm</label>
            <div className="grid grid-cols-2 gap-3">
              {MODELS.map(m => (
                <button
                  key={m.id}
                  onClick={() => setSelectedModel(m.id)}
                  className={`p-4 rounded-xl border text-left transition-all duration-300 relative overflow-hidden ${
                    selectedModel === m.id 
                      ? 'bg-brand-50 border-brand-500 text-brand-700 shadow-[0_0_15px_rgba(59,130,246,0.1)] ring-1 ring-brand-500' 
                      : 'bg-surface-50 border-surface-200 text-text-600 hover:border-surface-300 hover:bg-surface-100 hover:text-text-800'
                  }`}
                >
                  <div className={`font-semibold ${selectedModel === m.id ? 'text-brand-900' : 'text-text-800'}`}>{m.name}</div>
                  <div className={`text-xs mt-1 ${selectedModel === m.id ? 'text-brand-600' : 'text-text-500'}`}>{m.type}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <button
          onClick={handleTrain}
          disabled={training || !target}
          className="modern-button w-full py-4 flex items-center justify-center text-lg disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {training ? (
            <><RefreshCw className="mr-2 animate-spin" size={20} /> Starting Training...</>
          ) : (
            <><Play className="mr-2" size={20} /> Launch Experiment</>
          )}
        </button>
      </motion.div>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="flex items-center justify-between mb-4 border-b border-surface-200 pb-4">
          <h3 className="text-xl font-bold text-text-900">Experiment History</h3>
          <button onClick={fetchExperiments} className="text-text-500 hover:text-text-900 flex items-center text-sm bg-surface-50 px-3 py-1.5 rounded-lg border border-surface-200 transition-colors shadow-sm font-medium">
            <RefreshCw size={14} className="mr-2" /> Refresh
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <AnimatePresence>
            {experiments.map((exp, i) => (
              <motion.div
                layout
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                key={exp.id}
                onClick={() => exp.status === 'completed' && setSelectedExperimentId(exp.id)}
                className={`transition-all cursor-pointer group hover:-translate-y-1 ${
                  exp.status === 'completed' 
                    ? 'modern-card p-6 border-b-4 border-b-brand-500 hover:shadow-lg' 
                    : 'modern-card p-6 opacity-70'
                }`}
              >
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h4 className="font-bold text-text-900 mb-1 flex items-center">
                      <Box size={16} className="mr-2 text-brand-500" />
                      {exp.model_name}
                    </h4>
                    <span className="text-xs text-text-600 bg-surface-100 border border-surface-200 px-2 py-1 rounded-md font-medium">Target: {exp.target_column}</span>
                  </div>
                  <span className={`px-2.5 py-1 text-xs rounded-full flex items-center font-bold border ${
                    exp.status === 'completed' ? 'border-green-200 text-green-700 bg-green-50' :
                    exp.status === 'failed' ? 'border-red-200 text-red-700 bg-red-50' :
                    'border-brand-200 text-brand-700 bg-brand-50'
                  }`}>
                    {(exp.status === 'running' || exp.status === 'pending') && <Activity size={12} className="mr-1 animate-spin" />}
                    {exp.status.charAt(0).toUpperCase() + exp.status.slice(1)}
                  </span>
                </div>
                
                {exp.status === 'completed' && exp.metrics && (
                  <div className="grid grid-cols-2 gap-4 mt-6 pt-4 border-t border-surface-200">
                    {exp.metrics.task_type === 'regression' ? (
                      <>
                        <div>
                          <div className="text-xs text-text-500 font-semibold mb-1">R² Score</div>
                          <div className="text-lg font-bold text-text-900">{Number(exp.metrics.r2).toFixed(4)}</div>
                        </div>
                        <div>
                          <div className="text-xs text-text-500 font-semibold mb-1">RMSE</div>
                          <div className="text-lg font-bold text-text-900">{Number(exp.metrics.rmse).toFixed(4)}</div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div>
                          <div className="text-xs text-text-500 font-semibold mb-1">Accuracy</div>
                          <div className="text-lg font-bold text-text-900">{(exp.metrics?.accuracy * 100 || 0).toFixed(2)}%</div>
                        </div>
                        <div>
                          <div className="text-xs text-text-500 font-semibold mb-1">F1 Score</div>
                          <div className="text-lg font-bold text-text-900">{(exp.metrics?.f1_score * 100 || 0).toFixed(2)}%</div>
                        </div>
                      </>
                    )}
                  </div>
                )}
                {exp.status === 'failed' && (
                  <div className="mt-4 text-sm text-red-700 font-medium bg-red-50 p-3 rounded-lg border border-red-200">
                    Training Failed
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
}
