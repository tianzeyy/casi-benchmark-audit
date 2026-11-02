"""Refit all tested representations using an authorized local CASI copy."""
from pathlib import Path
import argparse,json,sys
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_revision import module,W

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-casi',type=Path,required=True)
    p.add_argument('--encoder-snapshot',type=Path);args=p.parse_args()
    out=W/'private/refit_results';out.mkdir(parents=True,exist_ok=True)
    b=module('baseline_refit',W/'legacy/baseline_core.py');b.CASI_DATA=args.source_casi
    examples,_,_=b.load_and_group_casi();split,audit,_=b.split_examples(examples)
    for name,rows in split.items():(out/('casi_'+name+'.json')).write_text(json.dumps([asdict(e) for e in rows]))
    reports={}
    for kind in ['word_tfidf','char_tfidf','combined_tfidf']:
        print('Fitting',kind,flush=True);models,info=b.train_family(kind,split['train'])
        rows=b.predict_records(models,split['test']);reports[kind]=dict(training=info,metrics=b.classification_metrics(rows))
        (out/(kind+'_test_predictions.json')).write_text(json.dumps(rows))
    (out/'tfidf_refit_summary.json').write_text(json.dumps(reports,indent=2))
    if args.encoder_snapshot:
        e=module('encoder_refit',W/'legacy/clinicalbert_core.py');e.RESULTS=out;e.MODEL_SNAPSHOT=args.encoder_snapshot;e.main()
    print('Refit outputs:',out)

if __name__=='__main__':main()
