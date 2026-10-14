"""Train CPU/GPU PyTorch experiments and select settings using validation only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import copy
import gc
import json
import random
import time
import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn
from dccad.data import load_reduced, prepare, windows, raw_episodes
from dccad.evaluation import calibrate, consecutive_alarm, metrics, incident_metrics, selection_key
from dccad.model import LSTMAutoencoder, predict_errors


def run(config, epochs, batch_size, seed, threads):
    torch.set_num_threads(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / "results" / config["name"]
    out.mkdir(parents=True, exist_ok=True)
    data = load_reduced(ROOT / "0.tar", ROOT / "artifacts" / f"reduced_{'enhanced' if config['enhanced'] else 'base'}.npz", config["enhanced"])
    if config.get('cleaning'):
        from dccad.preprocessing import prepare_clean
        data, scaled, split, scaler = prepare_clean(data, config.get('selected_features'), config.get('clip', 5.0))
    else:
        scaled, split, scaler = prepare(data, config.get("scaler", "standard"), config.get("clip", 5.0))
    width = config["width"]
    xtr, mtr = windows(data, scaled, split, 0, width, 4, True)
    xva, mva = windows(data, scaled, split, 1, width, 2)
    xte, mte = windows(data, scaled, split, 2, width, 2)
    if config.get('common_cohort_path'):
        cohort = np.load(ROOT / config['common_cohort_path'], allow_pickle=False)
        aligned = []
        for key, x, meta in [('train', xtr, mtr), ('validation', xva, mva), ('test', xte, mte)]:
            mask = np.isin(meta.row_index.to_numpy(), cohort[key])
            if int(mask.sum()) != len(cohort[key]):
                raise ValueError(f'{key} cohort is not fully covered by {config["name"]}')
            aligned.append((x[mask], meta[mask].reset_index(drop=True)))
        (xtr, mtr), (xva, mva), (xte, mte) = aligned
    epva, epte = raw_episodes(data, split, 1), raw_episodes(data, split, 2)
    norm = ~mva.y_window.to_numpy()
    xvn = xva[norm]
    model = LSTMAutoencoder(xtr.shape[2], width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tensor = torch.from_numpy(xtr)
    history, best_loss, best, stale = [], float("inf"), None, 0
    started = time.perf_counter()
    print(f"TRAIN {config['name']}: train={len(xtr):,}, val={len(xva):,}, test={len(xte):,}, device={device}",flush=True)
    for epoch in range(epochs):
        t0 = time.perf_counter()
        order = torch.randperm(len(tensor))
        model.train(); total = 0.
        for start in range(0, len(tensor), batch_size):
            x = tensor[order[start:start+batch_size]].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), x)
            loss.backward(); optimizer.step()
            total += loss.item() * len(x)
        model.eval(); val_total = 0.
        with torch.no_grad():
            for start in range(0, len(xvn), 1024):
                x = torch.from_numpy(xvn[start:start+1024]).to(device)
                val_total += nn.functional.mse_loss(model(x), x).item() * len(x)
        val_loss = val_total / len(xvn)
        history.append({"epoch":epoch+1,"train_loss":total/len(xtr),"val_loss":val_loss,"seconds":time.perf_counter()-t0})
        print(f"{config['name']} epoch {epoch+1}/{epochs}: loss={total/len(xtr):.5f}, val={val_loss:.5f}, {history[-1]['seconds']:.1f}s",flush=True)
        if val_loss < best_loss - 1e-6:
            best_loss, best, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(best)
    torch.save({"state_dict":{k:v.cpu() for k,v in model.state_dict().items()},
                "width":width,"features":data["features"].tolist(),"hidden":64,"latent":16}, out / "model.pt")
    joblib.dump(scaler, out / "scaler.joblib")
    pd.DataFrame(history).to_csv(out / "history.csv",index=False)
    va_mean, va_end, _ = predict_errors(model,xva,device=device)
    te_mean, te_end, te_feature = predict_errors(model,xte,device=device)
    policies = []
    for score_mode, va_score in [("whole_window",va_mean),("latest_reading",va_end)]:
        for q in [95.,99.,99.5]:
            for per_node in [False,True]:
                threshold = calibrate(mva[norm],va_score[norm],q,per_node)
                for persistence in [1,2]:
                    pred = consecutive_alarm(mva,va_score,threshold,persistence)
                    incidents,_ = incident_metrics(mva,pred,epva)
                    policies.append({"score_mode":score_mode,"quantile":q,"per_node":per_node,
                                     "persistence":persistence,"threshold":threshold,
                                     **metrics(mva.y_end,va_score,pred),**incidents})
    selected = max(policies,key=selection_key)
    va_score = va_mean if selected["score_mode"] == "whole_window" else va_end
    te_score = te_mean if selected["score_mode"] == "whole_window" else te_end
    pred = consecutive_alarm(mte,te_score,selected["threshold"],selected["persistence"])
    incidents,details = incident_metrics(mte,pred,epte)
    record = {**config,"feature_count":len(data['features']),"features":data['features'].tolist(),
              "seed":seed,"epochs_completed":len(history),"best_normal_val_loss":best_loss,
              "parameters":sum(p.numel() for p in model.parameters()),"device":device,
              "training_seconds":time.perf_counter()-started,"train_windows":len(xtr),
              "val_windows":len(xva),"test_windows":len(xte),"val_endpoint_positives":int(mva.y_end.sum()),
              "test_endpoint_positives":int(mte.y_end.sum()),"selected_policy":selected,
              "test":{**metrics(mte.y_end,te_score,pred),**incidents},
              "legacy_window_metrics":metrics(mte.y_window,te_mean,te_mean>np.percentile(va_mean[norm],99))}
    (out / "summary.json").write_text(json.dumps(record,indent=2))
    pd.DataFrame([{k:v for k,v in p.items() if k != 'threshold'} for p in policies]).to_csv(out / "validation_policies.csv",index=False)
    details.to_csv(out / "incident_details.csv",index=False)
    mte.to_csv(out / "window_metadata.csv",index=False)
    np.savez_compressed(out / "scores.npz",va_mean=va_mean,va_end=va_end,te_mean=te_mean,te_end=te_end,te_feature=te_feature)
    # Raw original readings remain separate from filled or clipped model inputs.
    dash={"node":mte.node.to_numpy(),"time_ns":mte.time_ns.to_numpy(),
          "label_end":mte.y_end.to_numpy().astype(np.int8),"label_window":mte.y_window.to_numpy().astype(np.int8),
          "score":te_score,"score_whole_window":te_mean,"score_latest":te_end,"alarm":pred.astype(np.int8)}
    raw = data["raw"][mte.row_index]
    reconstructed=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(xte),1024):
            reconstructed.append(model(torch.from_numpy(xte[start:start+1024]).to(device)).cpu().numpy()[:,-1])
    reconstructed_raw=scaler.inverse_transform(np.concatenate(reconstructed))
    for j,c in enumerate(data["features"]):
        dash[str(c)]=raw[:,j]
        dash[f"reconstructed_{c}"]=reconstructed_raw[:,j]
        dash[f"error_{c}"]=te_feature[:,j]
    pl.DataFrame(dash).write_parquet(out / "dashboard.parquet")
    node_rows=[]
    for n in mte.node.unique():
        m=mte.node.to_numpy()==n
        node_rows.append({"node":int(n),"samples":int(m.sum()),"positives":int(mte.y_end.to_numpy()[m].sum()),
                          **metrics(mte.y_end.to_numpy()[m],te_score[m],pred[m])})
    pd.DataFrame(node_rows).to_csv(out / "per_node.csv",index=False)
    print('COMPLETE',config['name'],json.dumps(record['test']),flush=True)
    del model,xtr,xva,xte,xvn,tensor,data,scaled
    gc.collect()
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--configs',nargs='+',default=['base24','base8','base48','enhanced24'])
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    configs={'base24':{'name':'base24','width':24,'enhanced':False},
             'base8':{'name':'base8','width':8,'enhanced':False},
             'base48':{'name':'base48','width':48,'enhanced':False},
             'enhanced24':{'name':'enhanced24','width':24,'enhanced':True},
             'robust24':{'name':'robust24','width':24,'enhanced':False,'scaler':'robust','clip':None}}
    records=[]
    for name in args.configs:
        records.append(run(configs[name],args.epochs,args.batch_size,args.seed,args.threads))
    best=max(records,key=lambda r:selection_key(r['selected_policy']))
    (ROOT/'results'/'selected_model.json').write_text(json.dumps({'name':best['name'],
          'criterion':'Highest endpoint F1 on labelled validation windows; ties use incident recall then false burst rate',
          'experiments':[r['name'] for r in records]},indent=2))
    print('SELECTED BY VALIDATION',best['name'],flush=True)


if __name__=='__main__':
    main()
