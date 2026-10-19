import subprocess
commands=[['/home/PengYanghan/miniconda3/envs/drone/bin/python', '/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/cache_gap_20261018/audit_records.py', '--launch', '/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/cache_gap_20261018/launch_v1.json'], ['/home/PengYanghan/miniconda3/envs/drone/bin/python', '/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/cache_gap_20261018/build_report.py', '--launch', '/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/cache_gap_20261018/launch_v1.json']]
for command in commands:
 subprocess.run(command,check=True)
