set -euo pipefail
root=/home/PengYanghan/Lunwen/chapter5-dynamic-service-20261028
src="$root/source_closed_loop_20261103"
v="$root/output/chapter5_dynamic_service_20261028/validation/queued_closed_loop_pilot_20261103"
py=/home/PengYanghan/miniconda3/envs/drone/bin/python
printf '%s  %s\n' '6d503704e3761c82fc926618b3e9a059916f638e7379e585975cbc1f5bc97b66' "$root/source_closed_loop_run_20261103.zip" | sha256sum -c -
cd "$src"
export LD_LIBRARY_PATH=/home/PengYanghan/miniconda3/envs/drone/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$py" -m unittest tests.test_queued_closed_loop tests.test_queued_root_advantage tests.test_queued_candidate_reliability tests.test_queued_joint_mpo > "$v/server_tests.log" 2>&1
"$py" -u scripts/analysis/queued_closed_loop_experiment.py --task-root "$root" --mode prepare > "$v/prepare.log" 2>&1
"$py" -u scripts/analysis/queued_closed_loop_experiment.py --task-root "$root" --mode smoke > "$v/smoke.log" 2>&1
"$py" -u scripts/analysis/queued_closed_loop_experiment.py --task-root "$root" --mode run > "$v/run.log" 2>&1
"$py" -u scripts/analysis/review_queued_closed_loop.py --task-root "$root" > "$v/server_review.log" 2>&1
"$py" -c "import pathlib,shutil;p=pathlib.Path('$v');shutil.copyfile(p/'local_review.json',p/'server_review.json')"
printf 'SUPERVISOR_COMPLETE\n'
