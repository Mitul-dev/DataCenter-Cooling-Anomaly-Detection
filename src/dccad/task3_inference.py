"""Load trusted Task 3 artifacts for standalone sensor-window scoring."""
from pathlib import Path
import json
import joblib
import numpy as np
import torch
from .supervised import LSTMClassifier,preprocess_csv,trailing_statistics
from .model import LSTMAutoencoder,predict_errors


def score_window(folder,frame):
    if 'timestamp' not in frame:
        raise ValueError('CSV inference requires timestamps to verify the 15-minute cadence')
    for field in ['node','node_id']:
        if field in frame and frame.tail(24)[field].nunique(dropna=False)!=1:
            raise ValueError('An input window must contain readings from only one server')
    folder=Path(folder)
    summary=json.loads((folder/'summary.json').read_text())
    pre=joblib.load(folder/'preprocessing.joblib')
    values=preprocess_csv(frame,pre)
    x=np.ascontiguousarray(values[-summary['width']:][None],dtype=np.float32)
    kind=summary['kind']
    if kind in ['lstm','autoencoder']:
        cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        model=(LSTMClassifier(len(cp['features']),cp['hidden'],cp['dropout']) if kind=='lstm'
            else LSTMAutoencoder(len(cp['features']),cp['width'],cp['hidden'],cp['latent']))
        model.load_state_dict(cp['state_dict']);model.eval()
        if kind=='lstm':
            with torch.no_grad():return {'logit':float(model(torch.from_numpy(x)).item())}
        mean,last,_=predict_errors(model,x)
        return {'whole_window':float(mean[0]),'latest_reading':float(last[0])}
    model=joblib.load(folder/'model.joblib')
    if kind=='boosting':return {'score':float(model.predict_proba(trailing_statistics(x))[0,1])}
    return {'anomaly_score':float(-model.score_samples(x.reshape(1,-1))[0])}
