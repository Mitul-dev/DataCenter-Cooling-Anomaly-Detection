"""Fresh inference for trusted additional-input recovery artifacts."""
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from .recovery import preprocess_recovery_csv
from .supervised import LSTMClassifier,trailing_statistics


def score_recovery_window(folder,frame):
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text())
    pre=joblib.load(folder/'preprocessing.joblib');values=preprocess_recovery_csv(frame,pre)
    x=np.ascontiguousarray(values[-8:][None],dtype=np.float32)
    if summary['kind']=='lstm':
        cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        model=LSTMClassifier(len(cp['features']),cp['hidden'],cp['dropout'])
        model.load_state_dict(cp['state_dict']);model.eval()
        with torch.no_grad():return float(model(torch.from_numpy(x)).item())
    model=joblib.load(folder/'model.joblib')
    return float(model.predict_proba(trailing_statistics(x))[0,1])
