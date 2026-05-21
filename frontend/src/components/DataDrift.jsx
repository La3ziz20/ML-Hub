import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, AlertTriangle, CheckCircle, Database, RefreshCw, ServerCrash } from 'lucide-react';

export default function DataDrift() {
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);
  const [iframeKey, setIframeKey] = useState(0);

  const fetchSummary = async () => {
    try {
      const res = await fetch('http://localhost:8000/drift/summary');
      const data = await res.json();
      if (data.status !== "none") {
        setSummary(data);
      }
    } catch (err) {
      console.error("Could not fetch drift summary:", err);
    }
  };

  useEffect(() => {
    fetchSummary();
  }, []);

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/drift/simulate', {
        method: 'POST'
      });
      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }
      const data = await res.json();
      setSummary(data);
      setIframeKey(k => k + 1); // reload iframe
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col h-[calc(100vh-80px)] pb-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-text-900 mb-1 flex items-center">
            <Activity className="mr-3 text-brand-500" /> Data Drift Monitoring
          </h2>
          <p className="text-text-500 text-sm max-w-3xl">
            Detect distribution shifts in your production data compared to the training baseline. We use Evidently AI and Kolmogorov-Smirnov tests to identify drifting features.
          </p>
        </div>
        <button 
          onClick={runSimulation}
          disabled={loading}
          className="btn-primary flex items-center"
        >
          {loading ? (
            <RefreshCw className="animate-spin mr-2" size={18} />
          ) : (
            <Database className="mr-2" size={18} />
          )}
          {loading ? 'Running Analysis...' : 'Simulate Drift'}
        </button>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 flex items-center">
          <ServerCrash className="mr-3" />
          <span>Error running simulation: {error}</span>
        </div>
      )}

      {summary && !loading && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div className="modern-card p-5 flex flex-col justify-center">
            <p className="text-text-500 text-sm font-semibold uppercase tracking-wider mb-1">Drift Share</p>
            <div className="flex items-end">
              <span className="text-4xl font-extrabold text-text-900">{(summary.drift_share * 100).toFixed(1)}%</span>
            </div>
            <p className="text-text-400 text-xs mt-2">Percentage of drifting columns</p>
          </div>

          <div className="modern-card p-5 flex flex-col justify-center">
            <p className="text-text-500 text-sm font-semibold uppercase tracking-wider mb-1">Drifted Columns</p>
            <div className="flex items-end">
              <span className="text-4xl font-extrabold text-text-900">{summary.drifted_columns}</span>
              <span className="text-xl font-bold text-text-400 ml-2 mb-1">/ {summary.total_columns}</span>
            </div>
            <p className="text-text-400 text-xs mt-2">Features showing significant shift</p>
          </div>

          <div className={`modern-card p-5 flex flex-col justify-center border-l-4 ${summary.dataset_drifted ? 'border-l-red-500 bg-red-50/30' : 'border-l-green-500 bg-green-50/30'}`}>
            <p className="text-text-500 text-sm font-semibold uppercase tracking-wider mb-2">Overall Status</p>
            <div className="flex items-center">
              {summary.dataset_drifted ? (
                <>
                  <AlertTriangle className="text-red-500 mr-3" size={32} />
                  <div>
                    <span className="text-xl font-bold text-red-700">Drift Detected</span>
                    <p className="text-red-600/80 text-sm">Consider retraining your model.</p>
                  </div>
                </>
              ) : (
                <>
                  <CheckCircle className="text-green-500 mr-3" size={32} />
                  <div>
                    <span className="text-xl font-bold text-green-700">Stable</span>
                    <p className="text-green-600/80 text-sm">No significant data drift.</p>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="flex-1 flex flex-col items-center justify-center bg-surface-50 rounded-xl border border-surface-200">
          <div className="animate-spin h-10 w-10 border-4 border-surface-200 border-t-brand-500 border-r-transparent rounded-full mb-4"></div>
          <h3 className="text-lg font-bold text-text-900">Analyzing Data Distributions...</h3>
          <p className="text-text-500">This might take 10-15 seconds for larger datasets.</p>
        </div>
      )}

      {!loading && summary && (
        <div className="flex-1 modern-card overflow-hidden flex flex-col">
          <div className="p-3 bg-surface-50 border-b border-surface-200 flex justify-between items-center">
            <h3 className="font-bold text-sm text-text-900 uppercase tracking-wide">Evidently AI Interactive Report</h3>
            <a href="http://localhost:8000/drift/report" target="_blank" rel="noreferrer" className="text-brand-600 hover:text-brand-700 text-xs font-semibold">
              Open Fullscreen ↗
            </a>
          </div>
          <div className="flex-1 bg-white relative">
            <iframe 
              key={iframeKey}
              src="http://localhost:8000/drift/report" 
              className="absolute inset-0 w-full h-full border-none"
              title="Evidently Report"
            />
          </div>
        </div>
      )}

      {!loading && !summary && (
        <div className="flex-1 flex flex-col items-center justify-center bg-surface-50 rounded-xl border border-surface-200 text-center px-4">
          <Activity size={64} className="text-surface-300 mb-6" />
          <h3 className="text-xl font-bold text-text-900 mb-2">No Drift Analysis Yet</h3>
          <p className="text-text-500 max-w-md">
            Click the "Simulate Drift" button to perturb your test set and run a comprehensive statistical comparison using Evidently AI.
          </p>
        </div>
      )}
    </motion.div>
  );
}
