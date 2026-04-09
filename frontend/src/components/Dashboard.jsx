import { useEffect, useState } from 'react';
import { getModels } from '../services/api';
import { Activity, Beaker, CheckCircle, XCircle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Dashboard() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModels().then(res => {
      setExperiments(res.data);
      setLoading(false);
    }).catch(err => {
      console.error(err);
      setLoading(false);
    });
  }, []);

  if (loading) return <div className="text-text-400 animate-pulse text-xl text-center mt-12 font-medium">Loading dashboard...</div>;

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <motion.div 
          whileHover={{ y: -2 }}
          className="modern-card p-4 flex flex-col justify-between"
        >
          <div className="text-text-500 text-xs font-semibold uppercase flex items-center mb-1"><Beaker size={14} className="mr-1.5 text-brand-500"/> Total Experiments</div>
          <div className="text-3xl font-bold text-text-900 leading-none">{experiments.length}</div>
        </motion.div>
        
        <motion.div 
          whileHover={{ y: -2 }}
          className="modern-card p-4 flex flex-col justify-between"
        >
          <div className="text-text-500 text-xs font-semibold uppercase flex items-center mb-1"><CheckCircle size={14} className="mr-1.5 text-green-500"/> Completed</div>
          <div className="text-3xl font-bold text-text-900 leading-none">
            {experiments.filter(e => e.status === 'completed').length}
          </div>
        </motion.div>
        
        <motion.div 
          whileHover={{ y: -2 }}
          className="modern-card p-4 flex flex-col justify-between"
        >
          <div className="text-text-500 text-xs font-semibold uppercase flex items-center mb-1"><Activity size={14} className="mr-1.5 text-brand-500"/> Processing</div>
          <div className="text-3xl font-bold text-text-900 leading-none">
            {experiments.filter(e => e.status === 'running' || e.status === 'pending').length}
          </div>
        </motion.div>
      </div>

      <div className="modern-card p-4">
        <div className="flex justify-between items-center border-b border-surface-200 pb-3 mb-3">
          <h2 className="text-lg font-bold text-text-900 flex items-center">Recent Activity</h2>
        </div>
        {experiments.length === 0 ? (
          <div className="text-text-400 py-6 text-center text-sm font-medium">
            No experiments found.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse table-auto">
              <thead>
                <tr className="text-text-500 border-b border-surface-200 text-xs uppercase tracking-wider font-semibold bg-surface-50">
                  <th className="py-2 px-3">Model</th>
                  <th className="py-2 px-3 text-center">Target Variable</th>
                  <th className="py-2 px-3 text-center">Status</th>
                  <th className="py-2 px-3 text-right">Primary Score</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                {experiments.slice(0, 10).map((exp, i) => {
                  let scoreLabel = '-';
                  if (exp.metrics) {
                    if (exp.metrics.task_type === 'regression') {
                       scoreLabel = <span title="R² Score"><span className="text-text-400 text-xs mr-2 border border-surface-200 px-1.5 py-0.5 rounded bg-surface-100 font-medium">R²</span>{Number(exp.metrics.r2).toFixed(4)}</span>;
                    } else {
                       scoreLabel = <span title="Accuracy"><span className="text-text-400 text-xs mr-2 border border-surface-200 px-1.5 py-0.5 rounded bg-surface-100 font-medium">ACC</span>{(exp.metrics.accuracy * 100).toFixed(1)}%</span>;
                    }
                  }
                  
                  return (
                  <motion.tr 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.05 }}
                    key={exp.id} 
                    className="border-b border-surface-100 text-text-800 hover:bg-surface-50 transition-colors text-sm"
                  >
                    <td className="py-2 px-3 font-semibold text-text-900">{exp.model_name}</td>
                    <td className="py-2 px-3 text-center"><span className="px-2 py-0.5 bg-surface-100 border border-surface-200 rounded text-xs text-text-600 font-medium">{exp.target_column}</span></td>
                    <td className="py-2 px-3 text-center">
                      <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium border ${
                        exp.status === 'completed' ? 'text-green-700 bg-green-50 border-green-200' :
                        exp.status === 'failed' ? 'text-red-700 bg-red-50 border-red-200' :
                        'text-brand-700 bg-brand-50 border-brand-200'
                      }`}>
                        {exp.status === 'running' && <Activity size={12} className="mr-1.5 animate-spin" />}
                        {exp.status.charAt(0).toUpperCase() + exp.status.slice(1)}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-right font-medium text-text-900">
                      {scoreLabel}
                    </td>
                  </motion.tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
