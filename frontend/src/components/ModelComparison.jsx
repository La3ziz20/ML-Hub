import { useEffect, useState } from 'react';
import { getMLflowRuns } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { Trophy, Activity, Target } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ModelComparison() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getMLflowRuns().then(res => {
      const completed = (res.data.runs || []).filter(r => r.status === 'FINISHED' && r.metrics && (r.metrics.accuracy || r.metrics.r2));
      
      const mapped = completed.map(r => {
         // Reconstruct the structure ModelComparison expects
         const isClassification = r.metrics.accuracy !== undefined;
         return {
            id: r.run_id.substring(0, 8),
            raw_id: r.run_id,
            model_name: r.parameters?.algo || r.run_id.substring(0,6),
            metrics: {
               task_type: isClassification ? 'classification' : 'regression',
               accuracy: Number(r.metrics.accuracy),
               f1_score: Number(r.metrics.f1_score),
               precision: Number(r.metrics.precision),
               recall: Number(r.metrics.recall),
               r2: Number(r.metrics.r2),
               rmse: Number(r.metrics.rmse),
               mse: Number(r.metrics.mse)
            }
         };
      });
      
      setModels(mapped);
      setLoading(false);
    });
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64 text-text-500 font-medium text-lg">
      <div className="animate-spin h-6 w-6 border-2 border-surface-200 border-t-brand-500 border-r-transparent rounded-full mr-4"></div>
      Loading comparison data...
    </div>
  );

  if (models.length === 0) return (
    <div className="text-center py-20 bg-surface-50 rounded-xl border border-surface-200">
      <Activity size={48} className="mx-auto text-text-300 mb-4" />
      <h3 className="text-xl font-bold text-text-900 mb-2">No Completed Models</h3>
      <p className="text-text-500">Train some models first to see them benchmarked here.</p>
    </div>
  );

  // Group models by classification vs regression
  const classModels = models.filter(m => m.metrics.task_type === 'classification');
  const regModels = models.filter(m => m.metrics.task_type === 'regression');

  const renderComparisonSection = (title, mData, type) => {
    if (mData.length === 0) return null;

    // determine metric to sort by to find "Best"
    const sortMetric = type === 'classification' ? 'accuracy' : 'r2';
    const sorted = [...mData].sort((a, b) => b.metrics[sortMetric] - a.metrics[sortMetric]);
    const bestModelId = sorted[0].id;

    // Prepare data for recharts
    const chartData = mData.map(m => {
       const d = { name: `${m.model_name} (#${m.id})`, isBest: m.id === bestModelId };
       if (type === 'classification') {
          d.Accuracy = m.metrics.accuracy * 100;
          d.F1_Score = m.metrics.f1_score * 100;
       } else {
          d.R2_Score = m.metrics.r2;
          d.RMSE = m.metrics.rmse;
       }
       return d;
    });

    return (
      <div className="mb-12">
         <h3 className="text-2xl font-bold text-text-900 flex items-center mb-6">
            <Target className="mr-3 text-brand-500" />
            {title}
         </h3>

         <div className="grid xl:grid-cols-12 gap-4 mb-6">
            <div className="modern-card p-4 xl:col-span-4 h-[350px] flex flex-col">
               <h4 className="font-bold text-sm text-text-700 mb-2 uppercase tracking-wide">Performance Metrics Chart</h4>
               <ResponsiveContainer width="100%" height="100%">
                 <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 50 }}>
                   <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                   <XAxis dataKey="name" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} angle={-25} textAnchor="end" />
                   <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                   <Tooltip 
                     cursor={{ fill: '#f8fafc' }} 
                     contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '0.75rem', padding: '12px' }}
                     formatter={(value) => (type === 'classification' ? value.toFixed(2) + '%' : value.toFixed(4))}
                   />
                   <Legend wrapperStyle={{ paddingTop: '20px' }} />
                   {type === 'classification' ? (
                     <>
                        <Bar dataKey="Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="F1_Score" fill="#ef4444" radius={[4, 4, 0, 0]} />
                     </>
                   ) : (
                     <>
                        <Bar dataKey="R2_Score" fill="#10b981" radius={[4, 4, 0, 0]} />
                     </>
                   )}
                 </BarChart>
               </ResponsiveContainer>
            </div>

            <div className="modern-card overflow-hidden xl:col-span-8 flex flex-col">
               <div className="p-3 bg-surface-50 border-b border-surface-200">
                  <h4 className="font-bold text-sm text-text-900 uppercase tracking-wide">Experiments Leaderboard</h4>
               </div>
               <div className="overflow-x-auto overflow-y-auto flex-1 max-h-[300px]">
                 <table className="w-full text-left text-xs whitespace-nowrap">
                    <thead className="bg-surface-50 text-text-500 uppercase tracking-wider sticky top-0 z-10 border-b border-surface-200 shadow-sm">
                        <tr>
                          <th className="px-4 py-2 font-semibold">Model</th>
                          {type === 'classification' ? (
                             <>
                               <th className="px-4 py-2 font-semibold text-right">Accuracy</th>
                               <th className="px-4 py-2 font-semibold text-right">F1 Score</th>
                               <th className="px-4 py-2 font-semibold text-right">Precision</th>
                               <th className="px-4 py-2 font-semibold text-right">Recall</th>
                             </>
                          ) : (
                             <>
                               <th className="px-4 py-2 font-semibold text-right">R² Score</th>
                               <th className="px-4 py-2 font-semibold text-right">RMSE</th>
                               <th className="px-4 py-2 font-semibold text-right">MSE</th>
                             </>
                          )}
                       </tr>
                    </thead>
                    <tbody className="divide-y divide-surface-200 text-text-800">
                       {sorted.map(m => (
                          <tr key={m.id} className={`hover:bg-surface-50 transition-colors ${m.id === bestModelId ? 'bg-brand-50/30' : ''}`}>
                             <td className="px-6 py-4 font-medium flex items-center">
                                {m.id === bestModelId && <Trophy size={16} className="text-yellow-500 mr-2" />}
                                {m.model_name} <span className="text-text-400 ml-2">#{m.id}</span>
                             </td>
                             {type === 'classification' ? (
                                 <>
                                  <td className={`px-4 py-2 text-right ${m.id === bestModelId ? 'font-bold text-brand-700' : ''}`}>{(m.metrics.accuracy * 100).toFixed(2)}%</td>
                                  <td className="px-4 py-2 text-right text-text-600">{(m.metrics.f1_score * 100).toFixed(2)}%</td>
                                  <td className="px-4 py-2 text-right text-text-600">{(m.metrics.precision * 100).toFixed(2)}%</td>
                                  <td className="px-4 py-2 text-right text-text-600">{(m.metrics.recall * 100).toFixed(2)}%</td>
                                </>
                             ) : (
                                <>
                                  <td className={`px-4 py-2 text-right ${m.id === bestModelId ? 'font-bold text-brand-700' : ''}`}>{m.metrics.r2.toFixed(4)}</td>
                                  <td className="px-4 py-2 text-right text-text-600">{m.metrics.rmse.toFixed(4)}</td>
                                  <td className="px-4 py-2 text-right text-text-600">{m.metrics.mse.toFixed(4)}</td>
                                </>
                             )}
                          </tr>
                       ))}
                    </tbody>
                 </table>
               </div>
            </div>
         </div>
      </div>
    );
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-text-900 mb-1">Model Comparison</h2>
        <p className="text-text-500 text-sm">Benchmark your completed models and identify the best performer.</p>
      </div>
      
      {renderComparisonSection('Classification Models', classModels, 'classification')}
      {renderComparisonSection('Regression Models', regModels, 'regression')}
      
    </motion.div>
  );
}
