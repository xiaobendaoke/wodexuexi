
import pathlib,json,pickle,numpy as np
root=pathlib.Path("/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/ranker_20261017/development_v1");data=np.load(root/'training_data.npz');layout=json.loads((root/'layout.json').read_text());folder=root/'folds/block_620001_workload_780001'
result={}
for name in ['ridge_actual','hgb_actual','hgb_predictor']:
 bundle=pickle.loads((folder/(name+'.pkl')).read_bytes());model=bundle['model'];scaler=bundle['scaler'];target=data['y_predictor'] if name.endswith('predictor') else data['y_actual']
 rows={}
 for split in ['train_states','test_states']:
  ids=bundle[split];total=den=oracle=0.;squared=variance=0.;weight_sum=0.
  for si in ids:
   z=layout[si]['costs']['10.0'];x=data['x'][z['start']:z['stop']];y=data['y_actual'][z['start']:z['stop']]
   scores=model.predict(scaler.transform(x) if scaler is not None else x);scores[z['actions'].index(-1)]=0.;action=max(range(len(scores)),key=lambda k:scores[k])
   total+=y[action];oracle+=max(y);den+=layout[si]['generated']
   tt=target[z['start']:z['stop']];ww=data['weight'][z['start']:z['stop']]
   squared+=float((ww*(scores-tt)**2).sum());variance+=float((ww*tt**2).sum());weight_sum+=ww.sum()
  rows[split]=dict(states=len(ids),gain_vs_G=total/den,oracle_gap=oracle/den,weighted_mse=squared/weight_sum,zero_predictor_mse=variance/weight_sum)
 result[name]=rows
(root/'analysis/fit_diagnostic.json').write_text(json.dumps(dict(fold=[620001,780001],no_refitting=True,diagnostic_only=True,models=result),indent=2)+'\n')
print(json.dumps(result))
