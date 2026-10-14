"""Score a timestamped recovery CSV without the original archive."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
import torch
from dccad.recovery_inference import score_recovery_window


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['extended_lstm8_p1','extended_boosting8'])
    p.add_argument('--csv',required=True,type=Path)
    args=p.parse_args();torch.set_num_threads(4)
    folder=ROOT/'results/task5_recovery_v1'/args.model
    score=score_recovery_window(folder,pd.read_csv(args.csv))
    print(json.dumps(dict(model=args.model,endpoint_score=score,
        interpretation='Uncalibrated current batch-client state score; not cooling-fault probability.',
        alarm='One endpoint score cannot establish a persistent alarm. Apply the saved policy to a gap-aware chronological score stream.'),indent=2))

if __name__=='__main__':main()
