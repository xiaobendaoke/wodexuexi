import importlib.util,sys,runpy
from pathlib import Path
spec=importlib.util.spec_from_file_location('framework_ready_resources','/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/extension_release_v2_20261001/utils/sf_hlm_resources.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
with m.CpuLease(Path('/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/train_logs/resources'),task=['framework_readiness_raw_audit'],kind='environment'):
 sys.argv=['/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/framework_readiness_20261018/audit_report.py', '--results', '/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/framework_readiness_20261018/diagnostic_v1', '--validation', '/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/framework_readiness_20261018']
 runpy.run_path('/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/framework_readiness_20261018/audit_report.py',run_name='__main__')
