"""DAI revision analyses. Original code and data are read-only reference inputs.

Run with an authorized CASI file to regenerate model predictions and identities.
The archived non-text predictions also support a source-free statistics rerun.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, platform, random, sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
import scipy, sklearn
from scipy import sparse
from sklearn.metrics import f1_score, roc_auc_score

W = Path(__file__).resolve().parents[1]
REF = W / "inputs/reference"
OUT = W / "results"
SEED, B = 20261030, 10000

def read(p):
    return json.loads(Path(p).read_text())

def write(p, data):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))

def csvout(p, rows):
    if not rows: return
    with Path(p).open("w", newline="", encoding="utf-8-sig") as f:
        d=csv.DictWriter(f, fieldnames=list(rows[0])); d.writeheader(); d.writerows(rows)

def module(name, p):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m)
    return m

def sid(e):
    return hashlib.sha256("\x1f".join([e.sf,e.sense,e.section,e.normalized_snippet]).encode()).hexdigest()

def norm(x): return " ".join(x.lower().split())

def scaled(p, t):
    z=np.log(np.clip(np.asarray(p,float),1e-12,1))/t; z-=z.max(); q=np.exp(z); return q/q.sum()

def decorate(records, t, gate):
    for r in records:
        q=scaled(r['probabilities'],t)
        r['scaled_probabilities']=q.tolist(); r['scaled_confidence']=float(q.max())
        r['confidence']=float(max(r['probabilities'])); r['pool']=len(q)
        r['passes']=bool(q.max()>=gate)
    return records

def score(models, examples):
    records=[]
    for sf in sorted({e.sf for e in examples}):
        pairs=[(i,e) for i,e in enumerate(examples) if e.sf==sf]
        model=models[sf]; classes=[str(c) for c in model.classes_]
        texts=[e.text for i,e in pairs]
        pp=model.predict_proba(texts); preds=model.predict(texts)
        for (i,e),p,pred in zip(pairs,pp,preds):
            records.append((i,dict(id=sid(e),sf=sf,gold=e.sense,prediction=str(pred),
                correct=bool(pred==e.sense),classes=classes,probabilities=p.tolist(),
                gold_index=classes.index(e.sense) if e.sense in classes else None)))
    return [r for _,r in sorted(records)]

def cluster_ratio(rows, value='passes', cluster='sf', reps=B):
    by=defaultdict(list)
    for r in rows: by[r[cluster]].append(float(r[value]))
    totals=np.array([sum(v) for v in by.values()]); sizes=np.array([len(v) for v in by.values()])
    rng=np.random.default_rng(SEED); vals=[]; n=len(sizes)
    for start in range(0,reps,200):
        z=rng.multinomial(n,np.repeat(1/n,n),size=min(200,reps-start))
        vals.extend(((z@totals)/(z@sizes)).tolist())
    return [float(x) for x in np.percentile(vals,[2.5,97.5])]

def calibration(rows, use_scaled):
    bins=[[] for _ in range(10)]; brier=[]
    for r in rows:
        p=r['scaled_probabilities'] if use_scaled else r['probabilities']; c=max(p)
        bins[min(9,int(c*10))].append((c,int(r['correct'])))
        brier.append(sum((v-int(i==r['gold_index']))**2 for i,v in enumerate(p)))
    reliability=[]; ece=0
    for i,b in enumerate(bins):
        n=len(b); mc=float(np.mean([x for x,y in b])) if n else None
        acc=float(np.mean([y for x,y in b])) if n else None
        if n: ece+=n/len(rows)*abs(mc-acc)
        reliability.append(dict(lower=i/10,upper=(i+1)/10,n=n,confidence=mc,accuracy=acc))
    return dict(n=len(rows),abbreviations=len({r['sf'] for r in rows}),ece=ece,
                brier=float(np.mean(brier)),bins=reliability)

def stratum(pool): return str(pool) if pool<5 else '5+'

def gate_summary(rows):
    n=len(rows); p=sum(r['passes'] for r in rows)
    return dict(n=n,abbreviations=len({r['sf'] for r in rows}),passing=p,rejected=n-p,
                pass_rate=p/n if n else None)

def support_auc(core,audit):
    sf=sorted({r['sf'] for r in core}|{r['sf'] for r in audit}); k=len(sf)
    cs=[[r['scaled_confidence'] for r in core if r['sf']==a] for a in sf]
    au=[[r['scaled_confidence'] for r in audit if r['sf']==a] for a in sf]
    n=np.array([len(x) for x in cs]); m=np.array([len(x) for x in au]); wins=np.zeros((k,k))
    for i,c in enumerate(cs):
        for j,a in enumerate(au):
            x=np.asarray(c)[:,None];y=np.asarray(a)[None,:]
            wins[i,j]=np.sum(x>y)+.5*np.sum(x==y)
    rng=np.random.default_rng(SEED); z=rng.multinomial(k,np.repeat(1/k,k),size=B)
    numerator=np.einsum('bi,ij,bj->b',z,wins,z,optimize=True)
    den=(z@n)*(z@m); values=numerator[den>0]/den[den>0]
    common=[i for i in range(k) if n[i] and m[i]]
    conditional=[float(wins[i,i]/(n[i]*m[i])) for i in common]
    return dict(positive_class='supported core test item',score='maximum temperature-scaled probability',
        pooled=float(wins.sum()/(n.sum()*m.sum())),cluster_ci=np.percentile(values,[2.5,97.5]).tolist(),
        common_abbreviations=len(common),within_abbreviation_macro=float(np.mean(conditional)),
        within_abbreviation_values=dict(zip([sf[i] for i in common],conditional)),
        common_abbreviation_pooled=float(roc_auc_score(
            [1]*sum(n[common])+[0]*sum(m[common]),
            [x for i in common for x in cs[i]]+[x for i in common for x in au[i]])))

def f1_bootstrap(preds):
    names=list(preds); ref=preds[names[0]]; gold=[r['sf']+'::'+r['gold'] for r in ref]
    labels=sorted(set(gold)); lookup={x:i for i,x in enumerate(labels)}; g=np.array([lookup[x] for x in gold])
    n=len(g);C=len(labels); groups=[np.flatnonzero(g==i) for i in range(C)]
    matrices={}; estimates={name:[] for name in names}
    for name,rs in preds.items():
        assert [(r['sf'],r['gold']) for r in rs]==[(r['sf'],r['gold']) for r in ref],name
        p=np.array([lookup[r['sf']+'::'+r['prediction']] for r in rs]); correct=p==g
        matrices[name]=(sparse.csr_matrix((np.ones(n),(np.arange(n),p)),shape=(n,C)),
            sparse.csr_matrix((correct.astype(float),(np.arange(n),g)),shape=(n,C)))
    sizes=np.bincount(g,minlength=C); rng=np.random.default_rng(SEED)
    for start in range(0,B,200):
        count=min(200,B-start); weights=np.zeros((count,n),dtype=np.int32)
        for ix in groups:
            weights[:,ix]=rng.multinomial(len(ix),np.repeat(1/len(ix),len(ix)),size=count)
        for name,(predmat,tpmat) in matrices.items():
            pc=weights@predmat;tp=weights@tpmat
            estimates[name].extend(np.mean(2*tp/(sizes+pc),axis=1).tolist())
    points={name:float(f1_score(gold,[r['sf']+'::'+r['prediction'] for r in rs],average='macro',labels=labels,zero_division=0)) for name,rs in preds.items()}
    return dict(method='10000 paired bootstrap draws stratified within 207 gold classes; fixed label list; zero_division=0',
        seed=SEED,classes=C,single_test_item_classes=sum(len(x)==1 for x in groups),
        models={name:dict(macro_f1=points[name],ci=np.percentile(estimates[name],[2.5,97.5]).tolist()) for name in names},
        differences={name+'_minus_word':dict(difference=points[name]-points['word_tfidf'],ci=np.percentile(np.asarray(estimates[name])-estimates['word_tfidf'],[2.5,97.5]).tolist()) for name in names if name!='word_tfidf'})

def generate_source_results(source):
    base=module('baseline_core',W/'legacy/baseline_core.py'); leak=module('leakage_core',W/'legacy/leakage_core.py')
    base.CASI_DATA=source; leak.CASI_DATA=source
    examples,conflicts,counts=base.load_and_group_casi(); split,audit,classes=base.split_examples(examples)
    print('Fitting reference word classifier on unchanged core split',flush=True)
    models,_=base.train_family('word_tfidf',split['train'])
    val=score(models,split['validation']); core=score(models,split['test']); rare=score(models,audit)
    old=read(REF/'word_tfidf_test_predictions.json')
    assert [(r['sf'],r['gold'],r['prediction']) for r in core]==[(r['sf'],r['gold'],r['prediction']) for r in old]
    maxdelta=max(abs(a-b) for r,o in zip(core,old) for a,b in zip(r['probabilities'],o['probabilities']))
    assert maxdelta<1e-7,maxdelta
    for partition,rs in [('core_test',core),('validation',val),('coverage_audit',rare)]: write(OUT/(partition+'_predictions.json'),rs)
    manifest=[dict(id=sid(e),partition=name) for name,es in list(split.items())+[('candidate_coverage_audit',audit)] for e in es]
    write(OUT/'split_manifest.json',dict(seed=20260722,support_threshold=5,records=manifest))
    raw=leak.load_raw_rows(); conventional=leak.partition(raw,5); test=conventional['test']; train=conventional['train']
    print('Fitting exposed and purged conventional-split arms',flush=True)
    exposed=leak.train_and_predict(train,test); identity_results={}
    for identity in ['reference','section_aware']:
        key=lambda e:(e.sf,e.normalized_snippet) if identity=='reference' else (e.sf,e.section,e.normalized_snippet)
        keys={key(e) for e in test}; trainkeys={key(e) for e in train}
        purged=[e for e in train if key(e) not in keys]; pred=leak.train_and_predict(purged,test)
        records=[]
        for e,a,b in zip(test,exposed,pred):
            records.append(dict(sf=e.sf,gold=e.sense,group=hashlib.sha256('\x1f'.join(key(e)).encode()).hexdigest(),
                exposed=key(e) in trainkeys,baseline_prediction=a['prediction'],purged_prediction=b['prediction'],
                baseline_correct=a['correct'],purged_correct=b['correct'],difference=int(a['correct'])-int(b['correct'])))
        write(OUT/(identity+'_leakage_predictions.json'),records)
        grouping=defaultdict(list)
        for e in raw: grouping[key(e)].append(e)
        retained=[v[0] for v in grouping.values() if len({e.sense for e in v})==1]
        conflict=[v for v in grouping.values() if len({e.sense for e in v})>1]
        tally=Counter((e.sf,e.sense) for e in retained)
        identity_results[identity]=dict(valid_raw_rows=len(raw),unique=len(retained),
            surplus_all_groups=sum(len(v)-1 for v in grouping.values()),
            nonconflict_surplus=sum(len(v)-1 for v in grouping.values() if len({e.sense for e in v})==1),
            conflict_groups=len(conflict),conflict_rows=sum(map(len,conflict)),
            core_classes=sum(v>=5 for v in tally.values()),audit_classes=sum(v<5 for v in tally.values()),
            core_contexts=sum(v for v in tally.values() if v>=5),audit_contexts=sum(v for v in tally.values() if v<5),
            purged_training_rows=len(train)-len(purged),exposed_test_rows=sum(r['exposed'] for r in records),
            baseline_purged_prediction_changes=sum(r['baseline_prediction']!=r['purged_prediction'] for r in records),
            candidate_classes_lost=len({(e.sf,e.sense) for e in train}-{(e.sf,e.sense) for e in purged}))
    write(OUT/'construction_counts.json',identity_results)
    write(OUT/'source_regeneration.json',dict(max_absolute_probability_difference=maxdelta,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),core_counts=counts,
        ms={k:v for k,v in classes.items() if k.startswith('MS::')},
        python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    # Contexts remain private. Clinical adjudication is deliberately left for the author.
    write(W/'private/audit_contexts.json',[dict(id=sid(e),sf=e.sf,gold=e.sense,section=e.section,context=e.snippet) for e in audit])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-casi',type=Path);a=parser.parse_args();OUT.mkdir(exist_ok=True)
    if a.source_casi: generate_source_results(a.source_casi)
    frozen=read(REF/'frozen_protocol.json')['k_sensitivity']['5']; t=frozen['temperature'];gate=frozen['validation_operating_point']['threshold']
    core=decorate(read(OUT/'core_test_predictions.json'),t,gate);audit=decorate(read(OUT/'coverage_audit_predictions.json'),t,gate)
    val=decorate(read(OUT/'validation_predictions.json'),t,gate)
    # Verify that the frozen gate is exactly the validation 90 percent operating point.
    sorted_conf=sorted((r['scaled_confidence'] for r in val),reverse=True)
    assert abs(sorted_conf[math.ceil(.9*len(val))-1]-gate)<1e-8
    assert sum(r['passes'] for r in audit)==131
    allgroups={'all_core':core,'correct_core':[r for r in core if r['correct']],
        'incorrect_core':[r for r in core if not r['correct']],'audit':audit}
    summary={'temperature':t,'threshold':gate,'bootstrap_seed':SEED,'bootstrap_replicates':B,
        'gate':{name:gate_summary(rs) for name,rs in allgroups.items()},'support_auroc':support_auc(core,audit)}
    strata=[]
    for s in ['1','2','3','4','5+']:
        group=[r for r in audit if stratum(r['pool'])==s]
        row=dict(pool=s,**gate_summary(group),mean_confidence=float(np.mean([r['scaled_confidence'] for r in group])),cluster_ci=cluster_ratio(group))
        for name in ['all_core','incorrect_core']:
            other=[r for r in allgroups[name] if stratum(r['pool'])==s]
            row[name]=gate_summary(other)
        strata.append(row)
    summary['audit_pool_strata']=strata
    cals={};calrows=[]
    for name,rs in [('all',core),('multiple_candidates',[r for r in core if r['pool']>1])]+[(f'pool_{s}',[r for r in core if stratum(r['pool'])==s]) for s in ['1','2','3','4','5+']]:
        cals[name]=dict(raw=calibration(rs,False),scaled=calibration(rs,True))
        calrows.append(dict(group=name,n=len(rs),abbreviations=len({r['sf'] for r in rs}),raw_ece=cals[name]['raw']['ece'],scaled_ece=cals[name]['scaled']['ece'],raw_brier=cals[name]['raw']['brier'],scaled_brier=cals[name]['scaled']['brier']))
    summary['calibration']=cals;csvout(OUT/'calibration_by_pool.csv',calrows)
    per_sf=[]
    for sf in sorted({r['sf'] for r in core}):
        rs=[r for r in core if r['sf']==sf];raw=calibration(rs,False);sc=calibration(rs,True)
        per_sf.append(dict(abbreviation=sf,n=len(rs),pool=rs[0]['pool'],raw_ece=raw['ece'],scaled_ece=sc['ece'],raw_brier=raw['brier'],scaled_brier=sc['brier']))
    csvout(OUT/'calibration_by_abbreviation.csv',per_sf)
    nonconstant=[r['scaled_ece'] for r in per_sf if r['pool']>1]
    summary['abbreviation_calibration_summary']=dict(n=len(nonconstant),median=float(np.median(nonconstant)),min=min(nonconstant),max=max(nonconstant),quartiles=np.percentile(nonconstant,[25,75]).tolist())
    preds={name:read(REF/(name+'_test_predictions.json')) for name in ['word_tfidf','char_tfidf','combined_tfidf','clinicalbert']}
    print('Computing paired macro F1 bootstrap intervals',flush=True)
    summary['macro_f1']=f1_bootstrap(preds)
    leakage={}
    for identity in ['reference','section_aware']:
        records=read(OUT/(identity+'_leakage_predictions.json')); parts={}
        for name,rs in [('all',records),('exposed',[r for r in records if r['exposed']]),('unexposed',[r for r in records if not r['exposed']])]:
            parts[name]=dict(n=len(rs),groups=len({r['group'] for r in rs}),difference_pp=100*float(np.mean([r['difference'] for r in rs])),
                cluster_ci_pp=[100*x for x in cluster_ratio(rs,'difference','group')])
        leakage[identity]=parts
    summary['leakage_cluster_sensitivity']=leakage
    # Complete, deterministically ordered author review packet; empty ratings are never counted as findings.
    contexts={r['id']:r for r in read(W/'private/audit_contexts.json')} if (W/'private/audit_contexts.json').exists() else {}
    packet=[]
    for i,r in enumerate(sorted([r for r in audit if r['passes']],key=lambda x:x['id'])):
        c=contexts.get(r['id'],{})
        packet.append(dict(item=i+1,id=r['id'],abbreviation=r['sf'],gold=r['gold'],prediction=r['prediction'],
            confidence=round(r['scaled_confidence'],8),candidate_meanings=' | '.join(r['classes']),
            context=c.get('context',''),relation='',rationale='',reference='',reviewer=''))
    if contexts:
        if not (W/'human_review/all_131_passing_items.csv').exists():
            csvout(W/'human_review/all_131_passing_items.csv',packet)
        sample=random.Random(SEED).sample(packet,50);sample.sort(key=lambda r:r['item'])
        if not (W/'human_review/random_sample_50.csv').exists():
            csvout(W/'human_review/random_sample_50.csv',sample)
        write(W/'human_review/sample_manifest.json',dict(population_n=131,sample_n=50,seed=SEED,ids=[r['id'] for r in sample]))
    write(OUT/'revision_summary.json',summary)
    write(OUT/'core_test_predictions.json',core);write(OUT/'coverage_audit_predictions.json',audit)
    print(json.dumps({k:summary[k] for k in ['gate','support_auroc','macro_f1','leakage_cluster_sensitivity']},indent=2),flush=True)

if __name__=='__main__': main()
