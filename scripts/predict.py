"""Score a chronological raw feature window with a saved model."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib,numpy as np,pandas as pd,torch
from dccad.model import LSTMAutoencoder
torch.set_num_threads(4)


def predict(folder,csv,node):
    summary=json.loads((folder/'summary.json').read_text())
    cp=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    frame=pd.read_csv(csv)
    if len(frame)<cp['width']:raise ValueError('Too few readings')
    if 'timestamp' in frame:
        times=pd.to_datetime(frame.timestamp.tail(cp['width']),utc=True)
        if not (times.diff().dropna()==pd.Timedelta(minutes=15)).all():raise ValueError('Invalid cadence')
    x=frame[cp['features']].tail(cp['width']).to_numpy(dtype=np.float32)
    if summary.get('cleaning'):
        from dccad.preprocessing import clean_inference
        x=clean_inference(x,cp['features'])
    if not np.isfinite(x).all():raise ValueError('Missing or nonfinite input')
    scaler=joblib.load(folder/'scaler.joblib');x=scaler.transform(x)
    if summary.get('clip',5.) is not None:x=np.clip(x,-summary.get('clip',5.),summary.get('clip',5.))
    model=LSTMAutoencoder(len(cp['features']),cp['width']);model.load_state_dict(cp['state_dict']);model.eval()
    with torch.no_grad():r=model(torch.from_numpy(x.astype(np.float32))[None]).numpy()[0]
    error=(x-r)**2;policy=summary['selected_policy']
    score=float(error.mean() if policy['score_mode']=='whole_window' else error[-1].mean())
    threshold=policy['threshold'];threshold=threshold[str(node)] if isinstance(threshold,dict) else threshold
    return {'score':score,'threshold':threshold,'above_threshold':score>threshold,
            'persistence_required':policy['persistence'],
            'top_latest_error_sensors':[cp['features'][i] for i in np.argsort(error[-1])[::-1][:3]]}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv',type=Path);p.add_argument('--model',default='selected');p.add_argument('--node',type=int,default=7)
    a=p.parse_args();name=json.loads((ROOT/'results'/'selected_model.json').read_text())['name'] if a.model=='selected' else a.model
    print(json.dumps(predict(ROOT/'results'/name,a.csv,a.node),indent=2))
