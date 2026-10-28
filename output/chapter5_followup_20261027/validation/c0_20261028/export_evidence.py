"""Seal C0 evidence, including failed engineering checks, on the server."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

parser=argparse.ArgumentParser();parser.add_argument('task_root',type=Path)
a=parser.parse_args();root=a.task_root.resolve()
if sys.platform!='linux':raise RuntimeError('Scientific evidence packaging runs on server')
result=root/'results/development_v2'
read=lambda p:json.loads(p.read_text())
assert read(result/'analysis/summary.json')['status']=='COMPLETE_C0_DEVELOPMENT'
assert read(result/'analysis/raw_audit.json')['status']=='PASS'
assert read(root/'output/validation/c0_tradeoff.collision.json')['summary']['fail']==0

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for z in iter(lambda:f.read(1024*1024),b''):h.update(z)
    return h.hexdigest()

members=['scripts','tests','results','output/validation']
files={}
for member in members:
    for path in (root/member).rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.name in ['archive_manifest.json','archive_receipt.json']:continue
        files[str(path.relative_to(root))]=dict(size=path.stat().st_size,sha256=sha(path))
manifest=dict(status='PASS',files=files,
    external_inputs='Original model/code hashes in protocol; reused B-stage observed states remain in B-stage evidence archive, not duplicated here',
    scope='C0 evidence and complete source; C1/training not started')
(root/'output/validation/archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=root/'output/c0_evidence_full.tar.gz'
if archive.exists():raise FileExistsError('Keep previous sealed archives')
def filter_member(info):
    return None if '__pycache__' in Path(info.name).parts else info
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for member in members:tar.add(root/member,arcname=member,filter=filter_member)
receipt=dict(status='SEALED',archive=str(archive),size=archive.stat().st_size,sha256=sha(archive),
    manifest_sha256=sha(root/'output/validation/archive_manifest.json'),files=len(files))
(root/'output/validation/archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
