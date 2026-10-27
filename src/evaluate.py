"""Summarize accuracy, macro-F1, invalid response rate, and paired consistency."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', default='results/predictions.csv')
    args = ap.parse_args()
    df = pd.read_csv(args.predictions, keep_default_na=False, dtype={'sample_id':str})
    labels = sorted(df.label.unique())
    clean = df[df.degradation=='clean'][['sample_id','prediction']].rename(columns={'prediction':'clean_prediction'})
    df = df.merge(clean, on='sample_id', how='left', validate='many_to_one')
    df['consistent_with_clean'] = df.prediction == df.clean_prediction
    rows = []
    for (kind, sev), g in df.groupby(['degradation','severity']):
        rows.append({'degradation':kind,'severity':sev,'n':len(g),'accuracy':g.correct.mean(),
                     'macro_f1':f1_score(g.label,g.prediction,labels=labels,average='macro',zero_division=0),
                     'valid_rate':g.valid.mean(), 'clean_consistency':g.consistent_with_clean.mean()})
    result = pd.DataFrame(rows).sort_values(['degradation','severity'])
    output = Path(args.predictions).parent
    result.to_csv(output/'metrics.csv',index=False)
    for kind,g in result[result.degradation!='clean'].groupby('degradation'):
        plt.plot(g.severity,g.accuracy,marker='o',label=kind)
    baseline = result[result.degradation=='clean']
    if len(baseline): plt.axhline(baseline.accuracy.iloc[0],linestyle='--',label='clean baseline')
    plt.xlabel('Severity (1–5)');plt.ylabel('Accuracy');plt.ylim(0,1);plt.xticks([1,2,3,4,5]);plt.legend(fontsize=8);plt.tight_layout()
    plt.savefig(output/'robustness_curves.png',dpi=200)
    print(result.to_string(index=False))

if __name__=='__main__':main()
