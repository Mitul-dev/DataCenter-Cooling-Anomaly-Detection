"""Score a trusted AHU model from an operational one-minute sensor window."""
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from .hvac import uploaded_features
from .supervised import LSTMClassifier,trailing_statistics


def score_window(folder,frame):
    folder=Path(folder);r=json.loads((folder/'summary.json').read_text());pre=joblib.load(folder/'preprocessing.joblib')
    x=np.ascontiguousarray(uploaded_features(frame,pre)[None],dtype=np.float32)
    if r['kind']=='lstm':
        cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        model=LSTMClassifier(len(cp['features']),cp['hidden'],cp['dropout']);model.load_state_dict(cp['state_dict']);model.eval()
        with torch.no_grad():return float(model(torch.from_numpy(x)).item())
    model=joblib.load(folder/'model.joblib');return float(model.predict_proba(trailing_statistics(x))[0,1])
