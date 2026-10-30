"""Execution-only acceleration for a separately frozen scientific source.

No new samples, PPO epochs, optimizer steps, or changed minibatches. Graph
backward passes within one existing minibatch run on shared immutable weights;
the parent alone clips gradients and steps Adam after every result arrives.
"""
import atexit
import multiprocessing as mp
import os
import queue
import time
import traceback


def _gradient_worker(policy, gradients, inputs, outputs, cpu):
    os.sched_setaffinity(0, {cpu})
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from marl_models.dynamic_service_hmarl import ppo_loss, tensor
    outputs.put(('ready', os.getpid(), cpu))
    while True:
        payload = inputs.get()
        if payload is None:
            return
        version, slot, item, divisor = payload
        try:
            policy.zero_grad(set_to_none=True)
            log, values, entropy = policy.replay(item['batch'], item['assignment'])
            target = torch.full_like(values, item['reward'])
            advantage = target-tensor(item['values'], policy.device)
            loss = ppo_loss(log, tensor(item['logprob'], policy.device), advantage, values, target, entropy)
            (loss/divisor).backward()
            used = []
            for parameter, buffer in zip(policy.parameters(), gradients[slot]):
                present = parameter.grad is not None
                used.append(present)
                if present:
                    buffer.copy_(parameter.grad)
                else:
                    buffer.zero_()
            outputs.put(('done', version, slot, float(loss.detach()), used))
        except BaseException:
            outputs.put(('error', version, slot, traceback.format_exc()))
            return


class ParallelLowerUpdater:
    """Persistent workers; weights are never updated while a graph is running."""
    def __init__(self, policy, cpus, timeout=180):
        import torch
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 4096 if hard == resource.RLIM_INFINITY else min(hard, 4096)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        if not cpus or len(set(cpus)) != len(cpus) or len(cpus) > 8:
            raise ValueError('one to eight distinct worker CPUs required')
        self.policy, self.cpus, self.timeout = policy, tuple(cpus), timeout
        self.version = 0
        self.context = mp.get_context('spawn')
        policy.share_memory()
        # Slots correspond to graph order, independently of worker scheduling.
        self.gradients = [[torch.zeros_like(p).share_memory_() for p in policy.parameters()] for _ in range(8)]
        self.inputs, self.outputs = self.context.Queue(), self.context.Queue()
        self.processes = [self.context.Process(target=_gradient_worker,
                          args=(policy, self.gradients, self.inputs, self.outputs, cpu), daemon=True)
                          for cpu in self.cpus]
        started = time.perf_counter()
        for process in self.processes:
            process.start()
        self.identities = []
        try:
            for _ in self.processes:
                message = self.outputs.get(timeout=timeout)
                if message[0] != 'ready':
                    raise RuntimeError(f'gradient worker startup failed: {message}')
                self.identities.append(dict(pid=message[1], cpu=message[2]))
        except BaseException:
            self.close()
            raise
        self.startup_seconds = time.perf_counter()-started
        self.last_timing = {}
        atexit.register(self.close)

    def update(self, optimizer, transitions):
        import numpy as np
        import torch
        valid = [item for item in transitions if len(item['batch'])]
        losses = []
        started = time.perf_counter()
        steps = 0
        for _ in range(4):
            for offset in range(0, len(valid), 8):
                minibatch = valid[offset:offset+8]
                self.version += 1
                for slot, item in enumerate(minibatch):
                    self.inputs.put((self.version, slot, item, len(minibatch)))
                results = {}
                for _ in minibatch:
                    try:
                        message = self.outputs.get(timeout=self.timeout)
                    except queue.Empty as exc:
                        raise RuntimeError('gradient worker did not finish; optimizer not stepped') from exc
                    if message[0] == 'error':
                        raise RuntimeError(message[-1])
                    if message[0] != 'done' or message[1] != self.version or message[2] in results:
                        raise RuntimeError('duplicate/stale gradient; optimizer not stepped')
                    results[message[2]] = message
                optimizer.zero_grad(set_to_none=True)
                for index, parameter in enumerate(self.policy.parameters()):
                    slots = [slot for slot in range(len(minibatch)) if results[slot][4][index]]
                    if slots:
                        parameter.grad = torch.zeros_like(parameter)
                        # Fixed reverse graph order mirrors backward traversal;
                        # numerical equivalence is tested rather than assumed.
                        for slot in reversed(slots):
                            parameter.grad.add_(self.gradients[slot][index])
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), .5)
                optimizer.step()
                steps += 1
                losses.extend(results[slot][3] for slot in range(len(minibatch)))
        self.last_timing = dict(seconds=time.perf_counter()-started, optimizer_steps=steps,
                                graphs=len(valid), workers=self.identities)
        return float(np.mean(losses)) if losses else 0.

    def close(self):
        processes = getattr(self, 'processes', [])
        if not processes:
            return
        for process in processes:
            if process.is_alive():
                self.inputs.put(None)
        for process in processes:
            if process.pid:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        self.processes = []
        # Release shared-memory file descriptors between validation pools and
        # remove the atexit reference that would otherwise retain all buffers.
        self.gradients = []
        self.policy = None
        for channel in (self.inputs, self.outputs):
            channel.cancel_join_thread()
            channel.close()
        atexit.unregister(self.close)


def install_environment_cache():
    """Cache pure current-slot construction only; no future state is cached."""
    from environment.dynamic_service import DynamicServiceEnv
    original_requests = DynamicServiceEnv.current_requests.fget
    original_batch = DynamicServiceEnv.batch

    def requests(self):
        cache = getattr(self, '_execution_requests_cache', None)
        if cache is None or cache[0] != self.t:
            cache = (self.t, original_requests(self))
            self._execution_requests_cache = cache
        return cache[1]

    def batch(self, positions=None, requests=None):
        if positions is not None or requests is not None:
            return original_batch(self, positions, requests)
        key = (self.t, self.positions.tobytes(), self.capacity.tobytes())
        cache = getattr(self, '_execution_batch_cache', None)
        if cache is None or cache[0] != key:
            cache = (key, original_batch(self))
            self._execution_batch_cache = cache
        return cache[1]

    DynamicServiceEnv.current_requests = property(requests)
    DynamicServiceEnv.batch = batch

    def restore():
        DynamicServiceEnv.current_requests = property(original_requests)
        DynamicServiceEnv.batch = original_batch
    return restore
