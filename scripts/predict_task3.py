"""Score one 24-reading CSV; persistent alarms require a chronological score stream."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
import torch
from dccad.task3_inference import score_window


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,help='Task 3 model folder name')
    parser.add_argument('--csv',required=True,type=Path)
    args=parser.parse_args();torch.set_num_threads(4)
    folder=ROOT/'results/task3_supervised'/args.model
    if folder.parent.resolve()!=(ROOT/'results/task3_supervised').resolve():raise ValueError('Invalid model folder')
    scores=score_window(folder,pd.read_csv(args.csv))
    print(json.dumps(dict(model=args.model,endpoint_scores=scores,
        interpretation='Uncalibrated current-state scores; not fault probabilities.',
        alarm='This command scores one endpoint. Apply the saved policy to a gap-aware score stream for persistence.'),indent=2))

if __name__=='__main__':main()
