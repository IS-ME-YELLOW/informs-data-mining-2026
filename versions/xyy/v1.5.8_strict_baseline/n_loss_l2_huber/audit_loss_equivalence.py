"""Explain observed equality using saved tree structures and conservative residual bounds."""
import lightgbm as lgb
from loss_protocol import *


def leaf_bounds(node):
    if 'leaf_value' in node:return float(node['leaf_value']),float(node['leaf_value'])
    left=leaf_bounds(node['left_child']);right=leaf_bounds(node['right_child'])
    return min(left[0],right[0]),max(left[1],right[1])


def main():
    data=load_data();rows=[];prefixes=[]
    saved=normalize(pd.read_parquet(RUN/'n_l2_oof_predictions.parquet'))
    oldpred=normalize(pd.read_parquet(HUBER/'base_predictions_cv.parquet'))
    for h in HORIZONS:
        target=component_target_name('N_t',h)
        y=data.component_targets[target].to_numpy();global_labels=y[np.isfinite(y)].astype(np.float32).astype(float)
        for outer in range(5):
            oldreceipt=read_json(HUBER/f'models/outer{outer}/{target}.json')
            receipt=read_json(RUN/f'models/outer{outer}/{target}.json')
            old=lgb.Booster(model_file=str(HUBER/oldreceipt['model_path']));new=lgb.Booster(model_file=str(RUN/receipt['model_path']))
            a=old.dump_model();b=new.dump_model();alpha=float(old.params['alpha'])
            assert old.params['objective']=='huber' and new.params['objective']=='regression'
            assert a['num_tree_per_iteration']==b['num_tree_per_iteration']==1
            assert not a['average_output'] and not b['average_output']
            structure_equal=a['tree_info']==b['tree_info']
            rounds_equal=oldreceipt['fit']['best_iterations']==receipt['fit']['best_iterations']
            assert structure_equal and rounds_equal
            hit=expected_mask(data.meta_train,h)&(data.row_folds==outer)
            diff=np.max(abs(saved.loc[hit,'raw_'+target].to_numpy()-oldpred.loc[hit,'raw_'+target].to_numpy()))
            assert diff==0
            # Serialized leaf outputs already contain shrinkage and the first-tree initial offset.
            # Summing independent leaf minima/maxima conservatively bounds every input path,
            # including combinations of leaves that no actual sample can simultaneously reach.
            low=high=0.;extreme=0.;lowest=0.;highest=0.
            for n,tree in enumerate(a['tree_info'],1):
                lo,hi=leaf_bounds(tree['tree_structure']);low+=lo;high+=hi
                bound=max(abs(low-global_labels.max()),abs(high-global_labels.min()))
                extreme=max(extreme,bound);lowest=min(lowest,low);highest=max(highest,high)
                prefixes.append(dict(horizon=h,outer_fold=outer,tree_prefix=n,prediction_lower_bound=low,
                    prediction_upper_bound=high,max_absolute_residual_bound=bound,alpha=alpha,bound_below_alpha=bound<alpha))
            assert extreme<alpha
            rows.append(dict(horizon=h,outer_fold=outer,Huber_alpha=alpha,global_label_min=float(global_labels.min()),
                global_label_max=float(global_labels.max()),saved_prefix_prediction_lower=lowest,saved_prefix_prediction_upper=highest,
                max_residual_bound=extreme,bound_below_alpha=True,tree_structures_identical=structure_equal,
                best_iterations_identical=rounds_equal,raw_OOF_max_abs_difference=float(diff),
                requested_rounds=receipt['fit']['requested_rounds'],Huber_model_sha256=oldreceipt['model_sha256'],
                L2_model_sha256=receipt['model_sha256']))
    write_frame(OUT/'summary/loss_equivalence.csv',rows)
    write_frame(OUT/'summary/saved_tree_prefix_bounds.csv',prefixes)
    result=dict(status='PASS',paired_models=20,tree_structures_identical=True,best_iterations_identical=True,
        raw_OOF_exactly_identical=True,Huber_alpha=.9,maximum_saved_prefix_residual_bound=max(r['max_residual_bound'] for r in rows),
        bound_covers='all inputs and label values in observed N range, for saved outer-refit tree prefixes',
        limitation='discarded inner-probe trees after best_iteration were not saved; their equality is not inferred from this bound',
        interpretation='Huber stays in its quadratic region on bounded saved trajectories; changing only objective to L2 yields identical trees',
        audit_code_sha256=sha256_file(Path(__file__)),training_performed=False)
    write_json(OUT/'summary/loss_equivalence_audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
