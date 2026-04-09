import { useEffect, useState } from 'react';
import { getMLflowRuns } from '../services/api';
import { Database, Clock, Settings, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

export default function MLflowHistory() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getMLflowRuns().then(res => {
      setRuns(res.data.runs || []);
      setLoading(false);
    });
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64 text-text-500 font-medium text-lg">
      <div className="animate-spin h-6 w-6 border-2 border-brand-200 border-t-brand-500 border-r-transparent rounded-full mr-4"></div>
      Connecting to MLflow Tracking Server...
    </div>
  );

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-text-900 mb-1 flex items-center">
            <Database className="mr-3 text-brand-500" />
            MLflow Native History
        </h2>
        <p className="text-text-500 text-sm">Traceable experiment logs directly queried from the python MLflow client tracking server.</p>
      </div>

      <div className="modern-card overflow-hidden">
         <div className="overflow-x-auto">
            <table className="w-full text-left text-sm whitespace-nowrap">
               <thead className="bg-surface-50 text-text-500 uppercase tracking-wider text-xs border-b border-surface-200">
                  <tr>
                     <th className="px-4 py-2 font-semibold">Run ID</th>
                     <th className="px-4 py-2 font-semibold">Status</th>
                     <th className="px-4 py-2 font-semibold">Parameters</th>
                     <th className="px-4 py-2 font-semibold">Metrics</th>
                     <th className="px-4 py-2 font-semibold">Artifacts URI</th>
                     <th className="px-4 py-2 font-semibold">End Time</th>
                  </tr>
               </thead>
               <tbody className="divide-y divide-surface-200 text-text-800">
                  {runs.map((r, i) => (
                     <tr key={i} className="hover:bg-surface-50 transition-colors border-b border-surface-100 last:border-none">
                        <td className="px-4 py-2 font-mono text-[11px] text-brand-600 font-bold">{r.run_id.substring(0,8)}...</td>
                        <td className="px-4 py-2 flex items-center">
                           <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${r.status === 'FINISHED' ? 'bg-green-100 text-green-700' : 'bg-surface-100 text-text-500'}`}>
                             {r.status}
                           </span>
                        </td>
                        <td className="px-4 py-2">
                           <div className="max-w-[200px] flex flex-wrap gap-1">
                              {Object.entries(r.parameters || {}).slice(0, 3).map(([k, v]) => (
                                 <span key={k} className="bg-surface-100 px-1.5 py-0.5 rounded text-[10px] border border-surface-200 truncate max-w-[120px]" title={`${k}: ${v}`}>
                                   <span className="font-semibold">{k}:</span> {v}
                                 </span>
                              ))}
                           </div>
                        </td>
                        <td className="px-4 py-2">
                           <div className="max-w-[250px] flex flex-wrap gap-1">
                              {Object.entries(r.metrics || {}).slice(0, 4).map(([k, v]) => (
                                 <span key={k} className="bg-brand-50 px-1.5 py-0.5 rounded text-[10px] border border-brand-100 text-brand-700">
                                   {k}: <span className="font-bold">{typeof v === 'number' ? v.toFixed(3) : v}</span>
                                 </span>
                              ))}
                           </div>
                        </td>
                        <td className="px-4 py-2 text-[10px] font-mono text-text-400 max-w-[150px] truncate" title={r.artifact_uri}>
                           {r.artifact_uri}
                        </td>
                        <td className="px-4 py-2 text-[11px] text-text-500">
                           {r.end_time ? new Date(r.end_time).toLocaleString() : 'N/A'}
                        </td>
                     </tr>
                  ))}
               </tbody>
            </table>
            {runs.length === 0 && (
               <div className="p-8 text-center text-text-400">No MLflow runs found.</div>
            )}
         </div>
      </div>
    </motion.div>
  );
}
