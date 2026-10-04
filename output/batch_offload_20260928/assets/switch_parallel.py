"""Checkpoint-verified switch of this experiment's own process group only."""
import hashlib,json,os,signal,subprocess,time,fcntl
from pathlib import Path
source=Path.cwd();root=source.parent
state_path=root/'pilot_controller_state.json'
smoke=json.loads((root/'parallel_smoke_state.json').read_text())
assert smoke['phase']=='VERIFICATION_COMPLETE'
old=json.loads(state_path.read_text());pid=old['pid']
assert old['phase']=='PILOT_RUNNING'
cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert str(root/'pilot_controller.py') in cmd
assert os.getpgid(pid)==pid
members=[]
for entry in Path('/proc').iterdir():
    if entry.name.isdigit():
        try:
            number=int(entry.name)
            if os.getpgid(number)==pid:
                command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                assert (entry/'cwd').resolve()==source,command
                members.append({'pid':number,'command':command})
        except (ProcessLookupError,FileNotFoundError):pass
assert len(members)==3,members
paused=False;terminated=False
try:
    os.killpg(pid,signal.SIGSTOP);paused=True
    time.sleep(.3)
    training=root/'pilot/main/lower_mlp_upper_uncovered_obs/seed_620001/train/attempt_0000'
    latest=json.loads((training/'latest.json').read_text());checkpoint=training/latest['checkpoint']
    contract=json.loads((checkpoint/'contract.json').read_text())
    assert contract['source_hash']==old['source_hash']
    for rel,sha in contract['files'].items():
        assert hashlib.sha256((checkpoint/rel).read_bytes()).hexdigest()==sha,rel
    saved={'prior_controller':old,'stopped_processes':members,'preserved_checkpoint':latest,
           'source_hash':contract['source_hash'],'changed_training_seeds':False,'threads_per_job':1,
           'training_workers':20,'evaluation_workers':1,'time_unix':time.time()}
    (root/'parallel_transition.json').write_text(json.dumps(saved,indent=2)+'\n')
    os.killpg(pid,signal.SIGTERM);os.killpg(pid,signal.SIGCONT);paused=False;terminated=True
    for _ in range(40):
        active=[]
        for member in members:
            path=Path(f'/proc/{member["pid"]}/stat')
            if path.exists() and path.read_text().split(') ',1)[1].split()[0]!='Z':active.append(member['pid'])
        if not active:break
        time.sleep(.25)
    assert not active,active
    with (root/'pilot/.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((training/'latest.json').read_text())==latest
    controller=root/'parallel_pilot_controller.py';controller.chmod(0o444)
    with (root/'pilot_parallel_controller.log').open('x') as log:
        process=subprocess.Popen(['bash',str(source/'run_remote_python.sh'),'-u',str(controller),'--source',str(source),
                                  '--output',str(root/'pilot'),'--state',str(state_path),'--workers','20'],
                                 cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(3)
    assert process.poll() is None,(root/'pilot_parallel_controller.log').read_text()[-2000:]
    saved.update(new_controller_pid=process.pid,controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),status='PARALLEL_STARTED')
    (root/'parallel_transition.json').write_text(json.dumps(saved,indent=2)+'\n')
    print(json.dumps(saved,indent=2))
finally:
    if paused and not terminated:os.killpg(pid,signal.SIGCONT)
