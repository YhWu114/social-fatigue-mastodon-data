"""Recompute dimension/ topic marginals and descriptive association effect sizes."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import chi2_contingency
from verify_release import rows,labels


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for source,field,prefix in [('data/fatigue_records.csv','fatigue_types','dimensions'),('data/topic_records.csv','topics','topics')]:
        data=rows(source);groups=sorted({x['language_group'] for x in data});types=sorted({t for x in data for t in labels(x[field])})
        totals=Counter(x['language_group'] for x in data);counts=Counter((x['language_group'],t) for x in data for t in labels(x[field]))
        with (args.output/(prefix+'_proportions.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['group','type','posts','denominator','percent'])
            for g in groups:
                for t in types:w.writerow([g,t,counts[g,t],totals[g],100*counts[g,t]/totals[g]])
        # Production association tests exclude posts with ANY unresolved topic judgment.
        known=[x for x in data if not x.get('unresolved_topics')]
        totals=Counter(x['language_group'] for x in known);counts=Counter((x['language_group'],t) for x in known for t in labels(x[field]))
        with (args.output/(prefix+'_effect_sizes.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['type','n','chi2','df','cramers_v','min_expected'])
            for t in types:
                table=np.array([[counts[g,t],totals[g]-counts[g,t]] for g in groups if totals[g]])
                chi,_,df,expected=chi2_contingency(table,correction=False)
                w.writerow([t,int(table.sum()),chi,df,np.sqrt(chi/table.sum()),expected.min()])
    print('Wrote proportions and effect sizes. Published bootstrap/simulation results remain in results/.')


if __name__=='__main__':
    main()
