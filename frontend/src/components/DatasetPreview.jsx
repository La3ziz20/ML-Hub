import { useEffect, useState, useRef } from 'react';
import { getDatasetPreview, uploadDataset, getPCA } from '../services/api';
import { Database, AlertCircle, Hash, Type, Upload, Activity } from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

export default function DatasetPreview() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [pcaData, setPcaData] = useState(null);
  const [loadingPca, setLoadingPca] = useState(false);
  const [pcaVariance, setPcaVariance] = useState([]);
  const fileInputRef = useRef(null);

  const fetchDataset = () => {
    setLoading(true);
    getDatasetPreview()
      .then(res => {
        setData(res.data);
        setLoading(false);
        setError(null);
      })
      .catch(err => {
        setError(err.response?.data?.detail || err.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchDataset();
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      await uploadDataset(formData);
      fetchDataset(); // Refresh table
    } catch (err) {
      alert("Failed to upload dataset: " + (err.response?.data?.detail || err.message));
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
         fileInputRef.current.value = '';
      }
    }
  };

  const handlePCA = async () => {
     setLoadingPca(true);
     try {
        const tgt = data?.columns ? data.columns[data.columns.length - 1] : null;
        const res = await getPCA(tgt);
        
        const formatted = res.data.pca1.map((val, i) => ({
            x: val,
            y: res.data.pca2[i],
            target: res.data.target && res.data.target.length > i ? res.data.target[i] : 'Unknown'
        }));
        
        const grouped = {};
        formatted.forEach(item => {
           const t = String(item.target);
           if (!grouped[t]) grouped[t] = [];
           grouped[t].push(item);
        });
        
        setPcaData(grouped);
        if(res.data.variance_ratio) {
           setPcaVariance(res.data.variance_ratio);
        }
     } catch(err) {
        alert("PCA failed: " + (err.response?.data?.detail || err.message));
     } finally {
        setLoadingPca(false);
     }
  };

  if (loading && !data) return (
    <div className="flex items-center justify-center h-64 text-text-500 font-medium text-lg">
      <div className="animate-spin h-6 w-6 border-2 border-surface-200 border-t-brand-500 border-r-transparent rounded-full mr-4"></div>
      Loading Dataset...
    </div>
  );
  
  if (error) return (
    <div className="bg-red-50 border border-red-200 text-red-700 p-6 rounded-2xl flex items-start">
      <AlertCircle className="mr-3 flex-shrink-0 mt-0.5" />
      <div>
        <h3 className="font-bold mb-1 text-lg">Failed to Load Dataset</h3>
        <p className="opacity-80">{error}</p>
        <button className="mt-4 modern-button px-4 py-2 text-sm" onClick={() => fileInputRef.current?.click()}>Upload a CSV</button>
        <input type="file" accept=".csv" className="hidden" ref={fileInputRef} onChange={handleFileUpload} />
      </div>
    </div>
  );

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-text-900 flex items-center mb-2">
            <Database className="mr-3 text-brand-500" />
            Dataset Preview
          </h2>
          <p className="text-text-500 text-sm">Total Records: <span className="text-text-900 font-bold">{data?.total_rows?.toLocaleString() || 0}</span></p>
        </div>
        <div>
          <div className="flex space-x-3">
            <button 
               onClick={handlePCA}
               disabled={loadingPca}
               className="bg-white border border-brand-200 text-brand-700 hover:bg-brand-50 px-4 py-2 rounded-xl flex items-center shadow-sm disabled:opacity-50 transition-colors font-medium cursor-pointer"
            >
              {loadingPca ? (
                 <div className="animate-spin h-4 w-4 border-2 border-brand-200 border-t-brand-500 rounded-full mr-2"></div>
              ) : (
                 <Activity size={18} className="mr-2" />
              )}
              {loadingPca ? 'Running PCA...' : 'Visualize Data (PCA)'}
            </button>
            <input type="file" accept=".csv" className="hidden" ref={fileInputRef} onChange={handleFileUpload} />
            <button 
               onClick={() => fileInputRef.current?.click()}
               disabled={uploading}
               className="modern-button px-4 py-2 flex items-center shadow-sm disabled:opacity-50 cursor-pointer"
            >
              {uploading ? (
                 <div className="animate-spin h-4 w-4 border-2 border-white/30 border-t-white rounded-full mr-2"></div>
              ) : (
                 <Upload size={18} className="mr-2" />
              )}
              {uploading ? 'Uploading...' : 'Upload CSV'}
            </button>
          </div>
        </div>
      </div>

      <div className="flex flex-col xl:flex-row gap-4">
      <AnimatePresence>
      {pcaData && (
        <motion.div 
          initial={{ opacity: 0, width: 0 }}
          animate={{ opacity: 1, width: '33.333333%' }}
          exit={{ opacity: 0, width: 0 }}
          className="modern-card p-4 shrink-0 overflow-hidden"
        >
           <h3 className="text-xl font-bold text-text-900 mb-2 flex items-center">
             <Activity className="mr-2 text-brand-500" />
             Dimensionality Reduction (2D PCA)
           </h3>
           {pcaVariance.length > 0 && (
             <p className="text-sm text-text-500 mb-6">
                Variance Explained: PC1: <span className="font-bold text-text-900">{(pcaVariance[0]*100).toFixed(1)}%</span>, PC2: <span className="font-bold text-text-900">{(pcaVariance[1]*100).toFixed(1)}%</span>
             </p>
           )}
           <div className="h-[400px] w-full bg-surface-50 rounded-xl p-4 border border-surface-200">
             <ResponsiveContainer width="100%" height="100%">
               <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                 <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                 <XAxis type="number" dataKey="x" name="PC1" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                 <YAxis type="number" dataKey="y" name="PC2" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 12 }} />
                 <Tooltip 
                   cursor={{ strokeDasharray: '3 3' }} 
                   contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '0.75rem', padding: '12px' }}
                 />
                 {Object.keys(pcaData).map((key, index) => {
                    const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
                    const color = colors[index % colors.length];
                    return (
                      <Scatter key={key} name={key} data={pcaData[key]} fill={color}>
                         {pcaData[key].map((entry, index) => (
                           <Cell key={`cell-${index}`} fill={color} />
                         ))}
                      </Scatter>
                    )
                 })}
               </ScatterChart>
             </ResponsiveContainer>
           </div>
           
           <div className="mt-4 flex flex-wrap gap-3 justify-center">
             {Object.keys(pcaData).map((key, index) => {
                const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
                const color = colors[index % colors.length];
                return (
                  <div key={key} className="flex items-center text-sm font-medium text-text-700 bg-surface-100 px-3 py-1.5 rounded-lg border border-surface-200">
                    <span className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: color }}></span>
                    Class: {key}
                  </div>
                )
             })}
           </div>
        </motion.div>
      )}
      </AnimatePresence>

      <div className={`modern-card flex-1 overflow-hidden transition-all duration-300`}>
        <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
          <table className="w-full text-left text-sm relative border-collapse">
            <thead className="sticky top-0 z-10 bg-surface-50 border-b border-surface-200 shadow-sm">
              <tr className="text-text-500">
                {data?.columns?.map((col, i) => (
                  <th key={i} className="px-3 py-2 font-semibold whitespace-nowrap text-xs border-r border-surface-200 last:border-r-0 uppercase tracking-wider bg-surface-50">
                    <div className="flex items-center">
                      <span className="opacity-70 mr-2 text-brand-500">
                        {data.preview && data.preview.length > 0 && typeof data.preview[0][col] === 'number' ? <Hash size={14} /> : <Type size={14} />}
                      </span>
                      {col}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="text-text-800">
              {data?.preview?.map((row, i) => (
                <tr key={i} className="hover:bg-surface-100 transition-colors group border-b border-surface-200 last:border-none text-xs">
                  {data?.columns?.map((col, j) => (
                    <td key={j} className="px-3 py-2 whitespace-nowrap group-hover:text-text-900 transition-colors border-r border-surface-100 last:border-r-0">
                      {typeof row[col] === 'number' ? (
                        <span className="text-brand-700 font-medium">{row[col]}</span>
                      ) : (
                        <span className="text-text-600">{row[col]}</span>
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="bg-surface-50 px-4 py-3 border-t border-surface-200 flex justify-between items-center text-xs text-text-500 font-medium">
           <span>Showing top <span className="text-text-900">{data?.preview?.length || 0}</span> records</span>
           <span><span className="text-text-900">{data?.columns?.length || 0}</span> Columns</span>
        </div>
      </div>
      </div>
    </motion.div>
  );
}
