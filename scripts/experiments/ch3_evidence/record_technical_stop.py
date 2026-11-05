from pathlib import Path
import json
base=Path('/home/PengYanghan/Lunwen/ch3-evidence-20261103')
root=base/'results';validation=base/'validation'
if list((root/'main').glob('*/seed_*/workload_*')):
 raise RuntimeError('Main confirmation evaluation exists; stop record needs further review')
state=json.loads((root/'status.json').read_text())
report=dict(status='UNIFORMLY_STOPPED_TECHNICAL_REVISION',reason='Delayed telemetry slot counts must describe received historical candidates rather than current transmitted candidates.',main_confirmation_evaluation_started=False,training_performance_used_for_restart_selection=False,seed_list_and_hypotheses_unchanged=True,all_cells_restart_from_scratch=True,initial_status=state,initial_training_episode_files=len(list((root/'main').glob('*/seed_*/train/episodes/*.json'))),original_release='release_v1',successor_release='release_v2',preserved_original_results=str(root))
path=validation/'technical_stop.json'
if path.exists():raise RuntimeError('Stop record already exists')
path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
