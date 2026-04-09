import { useEffect, useState } from 'react';
import { getModelDetails, predictModel } from '../services/api';
import { ChevronLeft, BarChart2, CheckCircle2, AlertTriangle, Target, Activity, Play, Box } from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar
} from 'recharts';
import { motion } from 'framer-motion';

export default function ResultsDashboard({ experimentId, onBack }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predictInputs, setPredictInputs] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    getModelDetails(experimentId).then(res => {
      setData(res.data);
      if (res.data.feature_columns) {
         const initial = {};
         res.data.feature_columns.forEach(c => initial[c] = '');
         setPredictInputs(initial);
      }
      setLoading(false);
    });
  }, [experimentId]);

  const handlePredict = async () => {
    setPredicting(true);
    try {
      const parsed = {};
      Object.keys(predictInputs).forEach(k => {
         parsed[k] = parseFloat(predictInputs[k]) || 0;
      });
      const res = await predictModel(data.id, parsed);
      setPredictionResult(res.data);
    } catch (err) {
      alert("Prediction failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setPredicting(false);
    }
  };

  if (loading) return <div className="text-text-400 font-medium animate-pulse text-xl text-center mt-12">Loading results...</div>;

  const metricsArray = data.metrics ? (
    data.metrics.task_type === 'regression' ? [
      { name: 'R² Score', value: data.metrics.r2, format: 'float' },
      { name: 'RMSE', value: data.metrics.rmse, format: 'float' },
      { name: 'MAE', value: data.metrics.mae, format: 'float' },
      { name: 'MSE', value: data.metrics.mse, format: 'float' },
    ] : [
      { name: 'Accuracy', value: data.metrics.accuracy, format: 'percent' },
      { name: 'Precision', value: data.metrics.precision, format: 'percent' },
      { name: 'Recall', value: data.metrics.recall, format: 'percent' },
      { name: 'F1 Score', value: data.metrics.f1_score, format: 'percent' },
    ]
  ) : [];

  const rocData = data.roc_curve ? data.roc_curve.fpr.map((fpr, i) => ({
    fpr: fpr,
    tpr: data.roc_curve.tpr[i]
  })) : [];

  return (
    <motion.div 
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="space-y-4 w-full"
    >
      <button 
        onClick={onBack}
        className="flex items-center text-text-500 hover:text-text-900 transition-colors mb-6 group bg-surface-50 px-4 py-2 rounded-xl border border-surface-200 w-fit shadow-sm font-medium"
      >
        <ChevronLeft size={20} className="mr-1 group-hover:-translate-x-1 transition-transform" />
        Back to Experiments
      </button>

      <div className="flex items-center justify-between mb-8 pb-4 border-b border-surface-200">
        <div>
          <h2 className="text-3xl font-bold text-text-900 mb-2 flex items-center">
            {data.model_name} <span className="text-text-400 text-lg ml-3 font-medium">#{data.id}</span>
          </h2>
          <div className="flex items-center space-x-4 text-sm font-medium">
            <span className="flex items-center text-text-600">
              <Target size={16} className="mr-1.5 text-brand-500" />
              Target: <span className="font-bold text-text-900 ml-1">{data.target_column}</span>
            </span>
            <span className="text-green-700 flex items-center bg-green-50 px-2.5 py-1 rounded-full border border-green-200">
              <CheckCircle2 size={14} className="mr-1.5" />
              Completed
            </span>
          </div>
        </div>
        
        <a 
          href={`http://localhost:8000/models/models/${data.id}/download`} 
          download 
          target="_blank" 
          rel="noreferrer"
          className="modern-button px-4 py-2 flex items-center shadow-sm text-sm"
        >
          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
          Download Model
        </a>
      </div>

      {data.metrics && (
        <div className="flex gap-4 mb-4">
          {metricsArray.map((m, i) => (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * i }}
              key={m.name} 
              className="modern-card p-3 flex-1 flex items-center justify-between"
            >
              <div className="text-text-500 text-xs font-bold uppercase tracking-wide">{m.name}</div>
              <div className="text-xl font-bold text-text-900">
                {m.format === 'percent' ? (
                  <>{(m.value * 100).toFixed(1)}<span className="text-sm text-text-400 ml-0.5">%</span></>
                ) : (
                  <>{Number(m.value).toFixed(4)}</>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}

      <div className="grid xl:grid-cols-3 gap-4">
        {data.confusion_matrix && (
          <div className="modern-card p-4 flex flex-col">
            <h3 className="text-xl font-semibold text-text-900 mb-6 flex items-center">
              <BarChart2 className="mr-2 text-brand-500" />
              Confusion Matrix
            </h3>
            <div className="grid gap-2">
              {data.confusion_matrix.map((row, i) => (
                <div key={i} className="flex gap-2">
                  {row.map((val, j) => {
                    const maxVal = Math.max(...data.confusion_matrix.flat());
                    const opacity = Math.max(0.05, Math.min(1, val / maxVal));
                    return (
                      <div 
                        key={j} 
                        className="flex-1 aspect-square rounded-xl flex items-center justify-center font-bold text-xl transition-all hover:scale-105 border"
                        style={{ 
                          backgroundColor: `rgba(59, 130, 246, ${opacity * 0.8})`,
                          color: opacity > 0.4 ? 'white' : '#1e3a8a',
                          borderColor: opacity > 0.1 ? 'transparent' : '#e2e8f0'
                        }}
                        title={`True: Class ${i}, Predicted: Class ${j}`}
                      >
                        {val}
                      </div>
                    )
                  })}
                </div>
              ))}
              <div className="text-center text-text-400 text-xs mt-auto bg-surface-50 rounded px-2 py-1 border border-surface-200 uppercase tracking-widest mt-4">Row: True | Col: Pred</div>
            </div>
          </div>
        )}

        {data.roc_curve && (
          <div className="modern-card p-4 flex flex-col">
            <h3 className="text-sm uppercase tracking-wide font-bold text-text-900 mb-1 flex items-center">
              <Activity size={16} className="mr-2 text-brand-500" />
              ROC Curve
            </h3>
            <p className="text-text-500 text-xs mb-4">Area Under Curve: <span className="text-text-900 font-bold">{data.roc_curve.auc.toFixed(4)}</span></p>
            <div className="flex-1 min-h-[250px] w-full mt-auto">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rocData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} tickLine={false} />
                  <YAxis type="number" domain={[0, 1]} stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', color: '#0f172a', borderRadius: '0.75rem', padding: '12px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)' }}
                    itemStyle={{ color: '#3b82f6', fontWeight: 'bold' }}
                    labelStyle={{ color: '#64748b', marginBottom: '4px' }}
                    formatter={(value) => [value.toFixed(4), "True Positive Rate"]}
                    labelFormatter={(label) => `False Pos Rate: ${label.toFixed(4)}`}
                  />
                  <Line type="monotone" dataKey="tpr" stroke="#3b82f6" strokeWidth={3} dot={false} activeDot={{ r: 6, fill: '#ffffff', stroke: '#3b82f6', strokeWidth: 2 }} animationDuration={1000} />
                  <Line type="monotone" data={[{fpr:0, tpr:0}, {fpr:1, tpr:1}]} dataKey="tpr" stroke="#94a3b8" strokeDasharray="5 5" strokeWidth={2} dot={false} activeDot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {data.metrics?.feature_importances && (
          <div className="modern-card p-4 flex flex-col">
            <h3 className="text-sm uppercase tracking-wide font-bold text-text-900 mb-4 flex items-center">
              <BarChart2 size={16} className="mr-2 text-brand-500" />
              Feature Importance
            </h3>
            <div className="flex-1 min-h-[250px] w-full flex flex-col">
              <ResponsiveContainer width="100%" height="100%">
                 <BarChart data={Object.entries(data.metrics.feature_importances).map(([k,v]) => ({name: k, Importance: v })).sort((a,b) => b.Importance - a.Importance).slice(0, 10)} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                   <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                   <XAxis type="number" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                   <YAxis dataKey="name" type="category" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} width={100} />
                   <Tooltip contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '0.75rem', padding: '12px' }} formatter={(val) => Number(val).toFixed(4)} />
                   <Bar dataKey="Importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} barSize={20} />
                 </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {data.feature_columns && data.feature_columns.length > 0 && (
        <div className="modern-card p-4 mt-4 bg-brand-50/20">
          <h3 className="text-xl font-bold text-text-900 mb-6 flex items-center">
            <Play className="mr-2 text-brand-500" />
            Live Prediction Sandbox
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
             {data.feature_columns.map(col => (
               <div key={col}>
                 <label className="block text-xs font-semibold text-text-600 mb-1 truncate" title={col}>{col}</label>
                 <input 
                   type="number" 
                   value={predictInputs[col]} 
                   onChange={e => setPredictInputs({...predictInputs, [col]: e.target.value})}
                   className="w-full bg-surface-50 border border-surface-300 rounded px-2 py-1 text-sm focus:border-brand-500 outline-none"
                   placeholder="0"
                 />
               </div>
             ))}
          </div>
          <div className="flex items-center space-x-6 bg-surface-50 p-3 rounded border border-surface-200 shadow-sm mt-4">
             <button 
               onClick={handlePredict} 
               disabled={predicting}
               className="modern-button px-4 py-2 shrink-0 disabled:opacity-50 text-sm"
             >
               {predicting ? 'Processing...' : 'Run Prediction'}
             </button>
             
             {predictionResult && (
               <div className="flex-1 flex items-center space-x-4 border-l border-surface-200 pl-4">
                 <div className="text-sm font-semibold text-text-500 uppercase tracking-wide">Result:</div>
                 <div className="text-2xl font-bold text-brand-600">
                    {typeof predictionResult.prediction === 'number' && !Number.isInteger(predictionResult.prediction) 
                       ? predictionResult.prediction.toFixed(4)
                       : predictionResult.prediction}
                 </div>
                 {predictionResult.probabilities && (
                   <div className="ml-auto text-xs text-text-500 bg-surface-100 px-3 py-1.5 rounded-lg border border-surface-200">
                     Probabilities: {JSON.stringify(predictionResult.probabilities.map(p => Number(p).toFixed(2)))}
                   </div>
                 )}
               </div>
             )}
          </div>
        </div>
      )}
    </motion.div>
  );
}
