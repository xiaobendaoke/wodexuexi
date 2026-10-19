import sys,json,pickle,gzip,math,collections,time
from pathlib import Path
from argparse import Namespace
sys.path.insert(0,"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/cache_calibration_b741d6faf839")
from scripts.analysis import sf_hlm_cache_calibration as c
a=Namespace(**{k:Path(v) for k,v in {"release_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/cache_calibration_b741d6faf839","feasibility_release":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/d448ad95f9a3","parent_launch":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/d448ad95f9a3/parent_launch.json","old_calibration":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/energy_price_20261017/calibration_v1","closed_loop":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/priced_closed_loop_20261017/development_v1","closed_validation":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/priced_closed_loop_20261017","results_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/cache_calibration_20261018/training_v2","validation_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/cache_calibration_20261018","resources":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/train_logs/resources","resource_module":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/extension_release_v2_20261001/utils/sf_hlm_resources.py"}.items()})
r=c.res(a)
with r.CpuLease(a.resources,task=['cache_branch_precision_diagnostic'],kind='environment'):
 f=c.load('cache_precision_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py');f.boot(a,620001)
 cc,mix=c.helpers(a)
 import config
 from environment import comm_model as comms
 p=c.read(a.results_root/'protocol.json');pw=c.read(a.results_root/'preparation_audit.json')['powers']
 def tx(size,rate):return float(size)*float(config.BITS_PER_BYTE)/max(float(rate),float(config.EPSILON))
 totals=collections.Counter();regimes={};maximum=dict(inverse=0.,direct=0.);failed_direct=[];public_edges=0
 out=a.validation_root/'physical_residuals_v2.json.gz'
 assert not out.exists()
 with gzip.open(out,'wt') as stream:
  stream.write('[');first=True
  for si,state in enumerate(p['states']):
   public=c.read(state['public']);labels=c.read(state['labels']);rows=public['rows']
   for row in rows:row['options']={int(k):v for k,v in row['options'].items()}
   env,obs=pickle.loads(Path(state['observed']).read_bytes());ues={u.id:u for u in env.ues};uavs={u.id:u for u in env.uavs};parts=[mix.decode_row(row,pw) for row in rows]
   exact={};truth={}
   for i,row in enumerate(rows):
    source=uavs[row['source']];ue=ues[row['ue_id']]
    upload=tx(row['size'],comms.calculate_ue_uav_rate(comms.calculate_channel_gain(ue.pos,source.pos),max(len(source.current_covered_ues),1)))
    for action,part in parts[i].items():
     if not cc.eligible(part):continue
     target=uavs[row['options'][action]['target']]
     hop=tx(row['size'],comms.calculate_uav_uav_rate(comms.calculate_channel_gain(source.pos,target.pos)))
     fetch=tx(config.FILE_SIZES[ue.current_request.req_id],comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(target.pos,config.MBS_POS)))
     exact[i,action]=(upload+hop,upload+hop+fetch);truth[i,action]=bool(target.cache[ue.current_request.req_id]);public_edges+=1
   for oi,outcome in enumerate(labels['outcomes'].values()):
    counts=collections.Counter(row['options'][action]['target'] for row,action in zip(rows,outcome['assignment']))
    for i,(row,action,request) in enumerate(zip(rows,outcome['assignment'],outcome['requests'])):
     if (i,action) not in exact:continue
     part=parts[i][action];hit=truth[i,action];target=row['options'][action]['target'];compute=part['work']*counts[target]
     inverse=(part['hit_base'] if hit else part['miss_base'])+compute
     direct=exact[i,action][0 if hit else 1]+compute;raw=float(request['latency'])
     ri=abs(raw-inverse);rd=abs(raw-direct);maximum['inverse']=max(maximum['inverse'],ri);maximum['direct']=max(maximum['direct'],rd)
     bad=ri>1e-6*(1+abs(raw));totals['checks']+=1;totals['inverse_failures']+=int(bad);totals['direct_failures']+=int(rd>1e-6*(1+abs(raw)))
     regime='one_minus_p_below_1e-8' if 1-part['p']<1e-8 else 'remaining'
     z=regimes.setdefault(regime,dict(checks=0,inverse_failures=0,max_inverse_residual=0.));z['checks']+=1;z['inverse_failures']+=int(bad);z['max_inverse_residual']=max(z['max_inverse_residual'],ri)
     record=dict(state_index=si,outcome_index=oi,row=i,action=action,p=part['p'],one_minus_p=1-part['p'],hit=hit,raw_latency=raw,inverse_latency=inverse,direct_latency=direct,inverse_residual=ri,direct_residual=rd,inverse_failed_fixed_tolerance=bad)
     if not first:stream.write(',')
     json.dump(record,stream,separators=(',',':'));first=False
  stream.write(']')
 report=dict(status='DIAGNOSED',checks=dict(totals),public_eligible_edges=public_edges,max_residual=maximum,by_conditioning=regimes,
  fixed_tolerance_unchanged=True,all_residuals_path=str(out),all_residuals_sha256=c.sha(out),source_protocol_sha256=c.sha(a.results_root/'protocol.json'),models_trained=0,
  interpretation='Finite precision inverse cache decomposition; direct predecision physical components are checked separately, no samples silently dropped')
 c.write(a.validation_root/'physical_precision_diagnostic.json',report);print(json.dumps(report),flush=True)
