import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import DatasetPreview from './components/DatasetPreview';
import ModelSelector from './components/ModelSelector';
import Dashboard from './components/Dashboard';
import ModelComparison from './components/ModelComparison';
import MLflowHistory from './components/MLflowHistory';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="dataset" element={<DatasetPreview />} />
          <Route path="experiments" element={<ModelSelector />} />
          <Route path="compare" element={<ModelComparison />} />
          <Route path="history" element={<MLflowHistory />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
