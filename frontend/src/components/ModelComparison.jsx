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
      // Only fetch regression runs (those with r2 metric)
      const completed = (res.data.runs || []).filter(r => r.status === 'FINISHED' && r.metrics && r.metrics.r2 !== undefined);
      
      const mapped = completed.map(r => {
         return {
            id: r.run_id.substring(0, 8),
            raw_id: r.run_id,
            model_name: r.parameters?.algo || r.run_id.substring(0,6),
            metrics: {
               task_type: 'regression',
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

  // All models are regression now
  const regModels = models;

  const renderComparisonSection = (title, mData) => {
    if (mData.length === 0) return null;

    // determine metric to sort by to find "Best"
    const sorted = [...mData].sort((a, b) => b.metrics.r2 - a.metrics.r2);
    const bestModelId = sorted[0].id;

    // Prepare data for recharts (Top 10 max to prevent clutter)
    const topModels = sorted.slice(0, 10);
    const displayNames = {
        "RandomForest": "Random Forest Regressor",
        "SVR": "Support Vector Regressor (SVR)",
        "KNN": "K-Nearest Neighbors Regressor",
        "Linear/Logistic": "Linear Regression",
        "AdaBoost": "AdaBoost Regressor",
        "XGBoost": "XGBoost Regressor"
    };

    const chartData = topModels.map(m => {
       const mappedName = displayNames[m.model_name] || m.model_name;
       const shortName = mappedName.replace(' Regressor', '').replace('Regression', '').replace(' (SVR)', '').trim();
       const d = { 
           name: `${shortName} #${m.id.substring(0,4)}`, 
           full_name: `${mappedName} (#${m.id})`,
           isBest: m.id === bestModelId 
       };
       // Visually clamp extreme negative R2 values so the chart isn't completely flattened,
       // but keep the real value for the tooltip.
       d.R2_Score = Math.max(-1, m.metrics.r2);
       d.real_R2 = m.metrics.r2;
       d.RMSE = m.metrics.rmse;
       return d;
    });

    return (
      <div className="mb-12">
         <h3 className="text-2xl font-bold text-text-900 flex items-center mb-6">
            <Target className="mr-3 text-brand-500" />
            {title}
         </h3>

         <div className="flex flex-col gap-6 mb-6">
            {/* Massive Full-Width Chart */}
            <div className="modern-card p-6 w-full h-[450px] flex flex-col shadow-md">
               <div className="flex items-center justify-between mb-4">
                 <h4 className="font-bold text-base text-text-900 uppercase tracking-wider flex items-center">
                    <Activity size={18} className="mr-2 text-brand-500" />
                    Performance Metrics Chart
                 </h4>
               </div>
               <ResponsiveContainer width="100%" height="100%">
                 <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                   <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                   <XAxis dataKey="name" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12, fontWeight: 500 }} angle={-25} textAnchor="end" interval={0} height={70} dy={15} />
                   <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 13, fontWeight: 500 }} />
                   <Tooltip 
                     cursor={{ fill: '#f1f5f9' }} 
                     contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '1rem', padding: '16px', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)' }}
                     labelFormatter={(label, payload) => payload[0] ? payload[0].payload.full_name : label}
                     formatter={(value, name, props) => props.payload.real_R2 !== undefined ? props.payload.real_R2.toFixed(4) : value.toFixed(4)}
                   />
                   <Legend verticalAlign="top" height={40} wrapperStyle={{ paddingBottom: '15px' }} iconType="circle" />
                   <Bar dataKey="R2_Score" radius={[8, 8, 0, 0]} maxBarSize={80} animationDuration={1500}>
                     {chartData.map((entry, index) => (
                       <Cell key={`cell-${index}`} fill={entry.isBest ? '#10b981' : '#cbd5e1'} className="transition-all duration-300 hover:opacity-80" />
                     ))}
                   </Bar>
                 </BarChart>
               </ResponsiveContainer>
            </div>

            {/* Leaderboard */}
            <div className="modern-card overflow-hidden w-full flex flex-col shadow-sm">
               <div className="p-4 bg-surface-50 border-b border-surface-200 flex items-center">
                  <Trophy size={18} className="mr-2 text-yellow-500" />
                  <h4 className="font-bold text-sm text-text-900 uppercase tracking-wide">Experiments Leaderboard</h4>
               </div>
               <div className="overflow-x-auto overflow-y-auto max-h-[400px]">
                 <table className="w-full text-left text-sm whitespace-nowrap">
                    <thead className="bg-surface-100 text-text-500 uppercase tracking-wider sticky top-0 z-10 border-b border-surface-200 shadow-sm">
                        <tr>
                          <th className="px-6 py-3 font-semibold">Model</th>
                          <th className="px-6 py-3 font-semibold text-right">R² Score</th>
                          <th className="px-6 py-3 font-semibold text-right">RMSE</th>
                          <th className="px-6 py-3 font-semibold text-right">MSE</th>
                       </tr>
                    </thead>
                    <tbody className="divide-y divide-surface-200 text-text-800">
                       {sorted.map((m, index) => {
                          const mappedName = displayNames[m.model_name] || m.model_name;
                          return (
                          <tr key={m.id} className={`hover:bg-surface-50 transition-colors ${m.id === bestModelId ? 'bg-brand-50/40' : ''}`}>
                             <td className="px-6 py-4 font-medium">
                                <div className="flex items-center">
                                    {m.id === bestModelId ? (
                                        <span className="bg-yellow-100 text-yellow-700 px-2.5 py-1 rounded-full text-xs font-bold mr-3 border border-yellow-200 flex items-center shrink-0">
                                           #1 Winner
                                        </span>
                                    ) : (
                                        <span className="text-text-400 font-bold w-10 shrink-0">#{index + 1}</span>
                                    )}
                                    <span className={`truncate ${m.id === bestModelId ? 'text-brand-900 font-bold' : ''}`}>{mappedName}</span> 
                                    <span className="text-text-400 ml-2 text-xs shrink-0">#{m.id}</span>
                                </div>
                             </td>
                             <td className={`px-6 py-4 text-right ${m.id === bestModelId ? 'font-bold text-brand-700 text-base' : ''}`}>{m.metrics.r2.toFixed(4)}</td>
                             <td className="px-6 py-4 text-right text-text-600">{m.metrics.rmse.toFixed(4)}</td>
                             <td className="px-6 py-4 text-right text-text-600">{m.metrics.mse.toFixed(4)}</td>
                          </tr>
                          );
                       })}
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
      
      {renderComparisonSection('Regression Models', regModels)}
      
    </motion.div>
  );
}
