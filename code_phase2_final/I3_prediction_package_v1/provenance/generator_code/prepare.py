"""Freeze inputs, provenance and all model tasks before any new fitting."""
from package_core import *
def main():
    sources={}
    def source(p,expected=None):
        p=Path(p).resolve();h=sha(p)
        if expected:assert h==expected,str(p)
        sources[str(p)]=h;return h
    index=read(BASE/'p_neighbor_horizon_increment/confirmation/summary/candidate_manifest.json')
    integration=BASE/'tree_stella_integration';sources0=read(integration/'reference/source_manifest.json')['files']
    for p,h in sources0.items():source(p,h)
    source(integration/'integration_config.json');source(integration/'reference/source_manifest.json')
    source(BASE/'protocol.py');source(BASE/'config.py')
    source(ROOT/'versions/stella_v112_strict/src/config.py');source(ROOT/'versions/stella_v112_strict/src/models.py')
    data=load_data();meta=pd.concat([data.meta_train[KEYS].assign(split='train'),data.meta_test[KEYS].assign(split='test')],ignore_index=True)
    for n,p in data.input_paths.items():source(p,data.loaded_hashes[n])
    frame(INPUT/'meta.parquet',meta)
    labels=pd.DataFrame({target(c,h):pd.concat([data.y_train[hname(h)] if c=='osi' else data.component_targets[target(c,h)],pd.Series(np.full(len(data.meta_test),np.nan))],ignore_index=True) for h in HS for c in ['osi',*CS]})
    frame(INPUT/'labels.parquet',labels)
    feat_sources={
        'base':[data.input_paths['features_train'],data.input_paths['features_test']],
        'p1':[BASE/'p_information_increment/features/v1'/f'F2_{s}.parquet' for s in ['train','test']],
        'p24':[BASE/'p_neighbor_horizon_increment/features/v1'/f'h24_{s}.parquet' for s in ['train','test']]}
    for key,pp in feat_sources.items():
        for p in pp:source(p)
        xs=[pd.read_parquet(p) for p in pp]
        for x,ref in zip(xs,[data.X_train,data.X_test]):pd.testing.assert_frame_equal(x[list(ref)],ref)
        assert all(len(x.columns)==(163 if key=='base' else 183) for x in xs);frame(INPUT/f'features_{key}.parquet',pd.concat(xs,ignore_index=True))
    raw=pd.concat([pd.read_csv(ROOT/f'data/DM_{s}.csv',usecols=['fipsCode','timestamp_et','P_t'],dtype={'fipsCode':str}) for s in ['Train','Test']],ignore_index=True)
    raw['hour']=((pd.to_datetime(raw.timestamp_et)-pd.Timestamp('2026-03-11'))/pd.Timedelta('1h')).astype(int);history=raw.loc[raw.hour<=71].copy()
    matrix=history.pivot(index='fipsCode',columns='hour',values='P_t');assert np.isfinite(matrix.to_numpy()).all()
    K=np.zeros(len(meta));s=meta.hour_idx.to_numpy()+1
    for hour in range(73,77):
        hit=s==hour;K[hit]=meta.fipsCode[hit].map(matrix.loc[:,list(range(hour-5,72))].sum(axis=1)/6).to_numpy()
    K[s>215]=np.nan;frame(INPUT/'known_history.parquet',meta[KEYS].assign(K1=K))
    # Complete observation-only history and original feature references travel with the package.
    frame(INPUT/'observed_P_history.parquet',history[['fipsCode','hour','P_t']])
    for seed in SEEDS:
        cv=ROOT/f'cv/cv_assignments_balanced_v1_seed{seed}.csv';source(cv);assignment=pd.read_csv(cv,dtype={'fipsCode':str}).set_index('fipsCode').fold
        fold=meta.fipsCode.map(assignment).fillna(-1).astype(int);assert set(fold[:34416])==set(FOLDS) and (fold[34416:]==-1).all()
        frame(INPUT/f'folds_{seed}.parquet',meta[KEYS].assign(fold=fold));copy_input(cv,INPUT/f'cv_seed{seed}.csv')
        tr=next(x for x in index['splits'] if x['split_seed']==seed)
        for n in ['component_predictions.parquet','control_predictions.parquet']:
            p=Path(tr['run_directory'])/n;source(p);copy_input(p,PORT/f'reference/split{seed}/tree_{n}')
        ip=integration/f'runs/split{seed}/integrated_predictions.parquet'
        if not ip.exists():
            candidates=list((integration/f'runs/split{seed}').glob('*predictions.parquet'));raise FileNotFoundError(str(ip)+'; available='+str(candidates))
        source(ip);copy_input(ip,PORT/f'reference/split{seed}/I3_candidates.parquet')
        st=ROOT/'versions/stella_v112_strict/runs'/f'stella_v112_nested_v1_split{seed}_model42'
        for n in ['candidate_predictions.parquet','thresholds.parquet']:source(st/n);copy_input(st/n,PORT/f'reference/split{seed}/stella_{n}')
    tasks=[]
    for seed in SEEDS:
        for s in [*(tuple([f]) for f in FOLDS),*SCOPES]:
            for name,r in RECIPES.items():
                if len(s)==1 and not (name.startswith('lgb_') and r['component'] in CS and r['horizon'] in (24,48)):continue
                t=dict(id=task_id(seed,s,name),seed=seed,scope=list(s),recipe=name,reuse=None)
                ref=reference_model(seed,s,r)
                if ref:
                    root,p=ref;rec=read(p);mp=root/rec['model_path'].replace('\\','/');actual=sha(mp)
                    hash_mode='raw'
                    if actual!=rec['model_sha256']:
                        assert r['family']=='lightgbm',str(mp)
                        rawbytes=mp.read_bytes();lf=rawbytes.replace(b'\r\n',b'\n')
                        if hashlib.sha256(lf).hexdigest()==rec['model_sha256']:hash_mode='git_crlf_canonical_lf'
                        else:
                            # LightGBM C++ writes LF; Windows Python appends only this footer as CRLF.
                            # Git normalizes the footer. Restore it in memory to authenticate the original hash.
                            point=lf.rfind(b'\npandas_categorical:');assert point>=0
                            restored=lf[:point]+lf[point:].replace(b'\n',b'\r\n')
                            assert hashlib.sha256(restored).hexdigest()==rec['model_sha256'],str(mp)
                            hash_mode='windows_python_footer_crlf_restored_in_memory'
                    source(mp);source(p)
                    scope=rec.get('task',{}).get('scope',rec.get('spec',{}).get('scope',rec.get('spec',{}).get('allowed_folds')))
                    assert scope==list(s),(p,scope,s)
                    feat=rec.get('feature_names',rec.get('spec',{}).get('feature_names'));assert feat==list(pd.read_parquet(INPUT/f'features_{r["feature"]}.parquet'))
                    t['reuse']=dict(model_path=str(mp),receipt_path=str(p),model_sha256=sha(mp),receipt_sha256=sha(p),original_model_id=rec['model_id'],upstream_model_sha256=rec['model_sha256'],hash_mode=hash_mode)
                tasks.append(t)
    assert len(tasks)==3396
    immutable_json(OUT/'tasks.json',tasks);immutable_json(OUT/'reference/source_hashes.json',sources)
    cfg=dict(protocol='i3_scope_prediction_v1',split_seeds=SEEDS,model_seed=42,recipes=RECIPES,scope_sizes=[2,3,4,5],singleton_anchor='FIPS sorted modulo 3 county groups; only LGB original components 24/48',
        component_formula=dict(zip(CS,[.4,.35,.25,-.1])),I3={'1':'T_C3','6':'T_C3','24':'H((T_C1+Stella_L2)/2)','48':'Stella_L2'},
        scalar_gate='positive cross-fitted LGB anchor q95 linear; L0>theta uses L0 else L1',max_rounds={name:maxrounds(r) for name,r in RECIPES.items()},
        patience=100,refit_rounds='floor mean probe best rounds, min1',workers=6,formal_GAT_training=False,submission=False,protocol_sha256=sha(OUT/'Protocol_2026-09-22.md'))
    immutable_json(OUT/'config.json',cfg)
    identity=dict(config_sha256=sha(OUT/'config.json'),task_manifest_sha256=sha(OUT/'tasks.json'),source_hashes=sources,
        input_hashes={str(p.relative_to(PORT)):sha(p) for p in sorted(INPUT.glob('*')) if p.is_file()},
        code_hashes={name:sha(OUT/name) for name in ['prepare.py','package_core.py','train_models.py']},environment=environment())
    immutable_json(OUT/'identity.json',dict(identity=identity,identity_hash=digest(identity)))
    reused=sum(t['reuse'] is not None for t in tasks);fits=sum((len(t['scope'])+1 if len(t['scope'])>1 else 4) for t in tasks if t['reuse'] is None)
    immutable_json(OUT/'preflight/budget.json',dict(total_models=len(tasks),reused_models=reused,new_models=len(tasks)-reused,new_fit_calls=fits,portable_contexts=3*(10+5+1)*4))
    print(dict(total_models=len(tasks),reused_models=reused,new_fit_calls=fits),flush=True)
if __name__=='__main__':main()
