import pathlib,shutil,subprocess,sys
r=pathlib.Path('/home/PengYanghan/Lunwen/chapter5-dynamic-service-20261028')
v=r/'output/chapter5_dynamic_service_20261028/validation/queued_closed_loop_pilot_20261103'
s=r/'audit_closed_loop_20261103'
x=subprocess.run([sys.executable,'-u',str(s/'scripts/analysis/review_queued_closed_loop.py'),'--task-root',str(s)],cwd=s)
if x.returncode:raise SystemExit(x.returncode)
shutil.copyfile(v/'local_review.json',v/'server_review.json')
