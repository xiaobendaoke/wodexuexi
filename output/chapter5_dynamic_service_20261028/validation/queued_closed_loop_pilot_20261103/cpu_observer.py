import json,os,time
from pathlib import Path
from collections import Counter
v=Path('/home/PengYanghan/Lunwen/chapter5-dynamic-service-20261028/output/chapter5_dynamic_service_20261028/validation/queued_closed_loop_pilot_20261103')
for _ in range(600):
 if (v/'run.lock').exists():break
 if (v/'run_failed.json').exists():raise SystemExit(0)
 time.sleep(1)
else:raise SystemExit('main PID not found')
pid=int((v/'run.lock').read_text());hz=os.sysconf('SC_CLK_TCK')
def sample():
 vals=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
 child=Path(f'/proc/{pid}/task/{pid}/children');ids=[int(x) for x in child.read_text().split()] if child.exists() else []
 cpus={};states=Counter()
 for p in ids+[pid]:
  try:
   text=Path(f'/proc/{p}/stat').read_text();s=text[text.rindex(')')+2:].split();cpus[p]=(int(s[11])+int(s[12]))/hz
   if p!=pid:states[s[0]]+=1
  except FileNotFoundError:pass
 return time.monotonic(),vals,cpus,dict(states)
old=sample();rows=[]
for _ in range(150):
 time.sleep(20);new=sample();dt=new[0]-old[0];delta=[b-a for a,b in zip(old[1],new[1])]
 own=sum(new[2][p]-old[2][p] for p in new[2].keys() & old[2].keys())/dt
 busy=(sum(delta)-delta[3]-delta[4])/hz/dt
 z=dict(epoch=time.time(),interval_seconds=dt,task_equivalent_cpus=own,global_busy_equivalent_cpus=busy,
  global_iowait_equivalent_cpus=delta[4]/hz/dt,global_steal_equivalent_cpus=delta[7]/hz/dt,
  busy_not_attributed_to_task=busy-own,worker_states=new[3],observed_children=len(new[2])-int(pid in new[2]))
 rows.append(z)
 (v/'cpu_context.json').write_text(json.dumps(dict(scope='Read-only concurrent machine context, not a replacement for measured task CPU accounting.',samples=rows),indent=2))
 print(json.dumps(z),flush=True);old=new
 if not Path(f'/proc/{pid}').exists() or (v/'run_complete.json').exists() or (v/'run_failed.json').exists():break
