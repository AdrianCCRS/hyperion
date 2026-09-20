"""Audita errores LOFO y abstenciones del candidato CPU congelado."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from fase2_clasificador.analysis.cpu_feature_strategies import production_variants
from fase2_clasificador.analysis.evaluate_cpu_feature_strategies import _fit, _prepare, _prototype, _weights
from fase2_clasificador.eval import protocol

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--metadata',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--n-jobs',type=int,default=1); p.add_argument('--seed',type=int,default=20260918); a=p.parse_args()
    meta=json.loads(a.metadata.read_text()); features=production_variants()['pmu_interactions']; threshold=float(meta['decision']['threshold'])
    d=_prepare(pd.read_csv(a.dataset,low_memory=False),1000,a.seed).dropna(subset=features).reset_index(drop=True); X=d[features].to_numpy(np.float32); y=d['_class'].to_numpy(bool); prob=np.full(len(d),np.nan)
    for tr,te,family in protocol.leave_one_kernel_out(d,kernel_col='kernel_family'):
        m=_prototype('xgboost',a.seed,y[tr],a.n_jobs,scale_pos_weight=1.0); _fit(m,X[tr],y[tr],_weights(d.iloc[tr])); prob[te]=m.predict_proba(X[te])[:,1]
    d['prob_memory']=prob; d['confidence']=np.maximum(prob,1-prob); d['prediction_memory']=prob>=.5; d['automatic']=d.confidence>=threshold; d['outcome']=np.where(~d.automatic,'revisar',np.where(d.prediction_memory==d._class,'correcta','error'))
    rows=[]
    for family,g in d.groupby('kernel_family',sort=True):
        s=g[g.automatic]; rows.append({'family':family,'n':len(g),'memory_share':float(g._class.mean()),'coverage':float(g.automatic.mean()),'f1_selected':float(f1_score(s._class,s.prediction_memory,average='macro',labels=[False,True],zero_division=0)) if len(s) else 0.,'error_auto_rate':float((s.prediction_memory!=s._class).mean()) if len(s) else 0.,'mean_confidence':float(g.confidence.mean())})
    a.output_dir.mkdir(parents=True,exist_ok=True); d.to_csv(a.output_dir/'cpu_selective_oof_predictions.csv',index=False); pd.DataFrame(rows).to_csv(a.output_dir/'cpu_selective_family_audit.csv',index=False)
    d.assign(freq_bin=pd.qcut(d.freq_khz_observed,4,duplicates='drop')).groupby(['freq_bin','outcome'],observed=True).size().rename('n').reset_index().to_csv(a.output_dir/'cpu_selective_frequency_outcomes.csv',index=False)
    print(f'rows={len(d)} threshold={threshold:.2f} automatic={d.automatic.mean():.3f}')
if __name__=='__main__': main()
