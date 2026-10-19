"""Linux CPU leases shared by experiment jobs and nested rollout workers.

No simulation dependencies: topology, locking and task identity can be tested
without Torch. A live adopted owner remains reserved after its supervisor exits.
"""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def _unsupported(value):
    raise TypeError('Unsupported JSON value: '+type(value).__name__)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x) if isinstance(x,Path) else _unsupported(x)) + '\n')
    temporary.replace(path)


def process_identity(pid):
    try:
        # The comm field can contain spaces; fields after its last ')' are safe.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, IndexError): return None


def topology(reserve=8):
    allowed = sorted(os.sched_getaffinity(0)); cores = {}
    for cpu in allowed:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        package = int((base / 'topology/physical_package_id').read_text())
        core = int((base / 'topology/core_id').read_text())
        nodes = list(base.glob('node[0-9]*')); node = int(nodes[0].name[4:]) if nodes else package
        cores.setdefault((node, package, core), []).append(cpu)
    groups = {}
    for key in sorted(cores): groups.setdefault(key[0], []).append(cores[key])
    ordered = []
    for index in range(max(map(len, groups.values()))):
        for node in sorted(groups):
            if index < len(groups[node]): ordered.append(groups[node][index])
    reserved = ordered[:reserve]; usable = ordered[reserve:]
    cpus = [siblings[t] for t in range(max(map(len, usable))) for siblings in usable if t < len(siblings)]
    return dict(allowed=allowed, physical_cores=len(ordered), reserved=[c for g in reserved for c in g],
                cpus=cpus, numa_nodes=len(groups), primary_slots=len(usable), maximum_slots=len(cpus))


class CpuLease:
    def __init__(self, root, *, owner=None, pin=True, timeout=3600, task=None,kind=None):
        self.root = Path(root); self.owner = owner or os.getpid(); self.pin = pin
        self.task = task; self.kind=kind;self.type_fd=None; self.timeout = timeout; self.fd = None; self.cpu = None

    def _type_permit(self,profile):
        if self.kind is None:return True
        limit=profile.get('concurrency_by_kind',{}).get(self.kind,profile['slots'])
        for i in range(limit):
            path=self.root/'kinds'/self.kind/f'{(i+self.owner)%limit}.lock';path.parent.mkdir(parents=True,exist_ok=True)
            fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:os.close(fd);continue
            self.type_fd=fd;return True
        return False

    def _release_type(self):
        if self.type_fd is not None:
            fcntl.flock(self.type_fd,fcntl.LOCK_UN);os.close(self.type_fd);self.type_fd=None

    def acquire(self):
        start = time.monotonic()
        while True:
            profile = json.loads((self.root / 'profile.json').read_text())
            if not self._type_permit(profile):
                if time.monotonic()-start>self.timeout:raise TimeoutError('Task-type CPU concurrency reached')
                time.sleep(.03);continue
            cpus = profile['cpus'][:profile['slots']]
            # All contenders rotate, avoiding a hot first-core lock convoy.
            primary=min(len(cpus),profile.get('primary_slots',profile.get('physical_cores',len(cpus))-len(profile.get('reserved',[]))//2))
            physical,secondary=cpus[:primary],cpus[primary:]
            shift=self.owner%len(physical)
            cpus=physical[shift:]+physical[:shift]+secondary
            for cpu in cpus:
                path = self.root / 'slots' / f'{cpu}.lock'; path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: os.close(fd); continue
                raw = os.read(fd, 2048)
                try: prior = json.loads(raw) if raw else {}
                except json.JSONDecodeError: prior = {}
                if prior.get('leased') and prior.get('pid') != self.owner and process_identity(prior['pid']) == prior.get('identity'):
                    fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd); continue
                self.fd, self.cpu = fd, cpu
                self._write(dict(leased=True, cpu=cpu, pid=self.owner, identity=process_identity(self.owner), acquired=time.time(),task=self.task,kind=self.kind))
                if self.pin: os.sched_setaffinity(self.owner, {cpu})
                return self
            self._release_type()
            if time.monotonic() - start > self.timeout: raise TimeoutError('No experiment CPU lease available')
            time.sleep(.03)

    def _write(self, value):
        os.lseek(self.fd, 0, 0); os.ftruncate(self.fd, 0)
        os.write(self.fd, json.dumps(value).encode())

    def release(self):
        if self.fd is not None:
            self._write(dict(leased=False, cpu=self.cpu, pid=self.owner))
            fcntl.flock(self.fd, fcntl.LOCK_UN); os.close(self.fd); self.fd = None
        self._release_type()

    def adopt_owner(self,pid):
        """Keep a spawned child reserved even if its dispatcher exits."""
        self.owner=pid
        self._write(dict(leased=True,cpu=self.cpu,pid=pid,identity=process_identity(pid),acquired=time.time(),task=self.task))

    @contextlib.contextmanager
    def suspended(self):
        self.release()
        try: yield
        finally: self.acquire()

    def __enter__(self): return self.acquire()
    def __exit__(self, *_): self.release()


@contextlib.contextmanager
def task_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Task already running: ' + str(path))
        yield


def trajectory_seed(*parts):
    payload = json.dumps(parts, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')


def initialize(root, slots=64):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    path = root / 'profile.json'
    if not path.exists():
        info = topology(); atomic_json(path, dict(**info, slots=min(slots, info['maximum_slots']), calibrated=False))
    return json.loads(path.read_text())


def telemetry(root, stop):
    """Append execution metadata; never rewrite scientific result contracts."""
    root = Path(root); last_cpu = None; last_time = None
    while not stop.is_set():
        values = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
        now = time.time(); total = sum(values); idle = values[3] + values[4]
        busy = None if last_cpu is None else 1 - (idle - last_cpu[1]) / max(total - last_cpu[0], 1)
        memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        leased = []
        for p in (root / 'slots').glob('*.lock'):
            try:
                d = json.loads(p.read_text())
                if d.get('leased') and process_identity(d['pid']) == d.get('identity'): leased.append(d)
            except (OSError, ValueError, KeyError): pass
        row = dict(unix=now, elapsed=None if last_time is None else now-last_time, cpu_busy_fraction=busy,
                   load_average=os.getloadavg(), available_memory_kib=memory['MemAvailable'], active_leases=leased,
                   disk_io=Path('/proc/diskstats').read_text(), profile=json.loads((root/'profile.json').read_text()))
        with (root / 'telemetry.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
        last_cpu, last_time = (total, idle), now
        stop.wait(30)
