import httpx
import time
import json

target = 'Category'
models = ['RandomForest', 'KNN', 'Linear/Logistic', 'SVM']
exp_ids = []

for m in models:
    res = httpx.post('http://localhost:8000/models/train', json={'model_name': m, 'target_column': target})
    data = res.json()
    print('Started', m, data)
    exp_ids.append(data['id'])

print('Waiting for completion...')
completed = set()
while len(completed) < len(exp_ids):
    for eid in exp_ids:
        if eid not in completed:
            try:
                r = httpx.get(f'http://localhost:8000/models/models/{eid}')
                state = r.json()
                if state.get('status') in ['completed', 'failed']:
                    print(f'Experiment {eid} finished with status {state.get("status")}')
                    if state.get('status') == 'failed':
                        print('Error:', state.get('error_message'))
                    else:
                        print(m, 'Metrics:', json.dumps(state.get('metrics'), indent=2))
                    completed.add(eid)
            except Exception as e:
                pass
    time.sleep(5)
