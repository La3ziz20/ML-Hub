import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:8000',
});

export const getDatasetPreview = () => api.get('/dataset/');
export const uploadDataset = (formData) => api.post('/dataset/upload', formData, {
    headers: {
        'Content-Type': 'multipart/form-data'
    }
});
export const trainModel = (data) => api.post('/models/train', data);
export const getModels = () => api.get('/models/models');
export const getModelDetails = (id) => api.get(`/models/models/${id}`);
export const predictModel = (id, features) => api.post(`/models/models/${id}/predict`, { features });
export const getPCA = (targetColumn) => api.get(`/dataset/pca${targetColumn ? `?target_column=${targetColumn}` : ''}`);
export const getMLflowRuns = () => api.get(`/mlflow/runs`);

export default api;
