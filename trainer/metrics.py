"""Local, resumable metrics with explicit timing and CUDA allocator measurements."""
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist


class MetricWriter:
    def __init__(self, path, metadata, resume_step=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.elapsed_seconds = 0.0
        if resume_step is not None and self.path.exists():
            rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
            rows = [row for row in rows if row.get('step', row.get('update', 0)) <= resume_step]
            self.elapsed_seconds = rows[-1].get('elapsed_seconds', 0.) if rows else 0.
            self.path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
        elif resume_step is None:
            self.path.write_text('')
        self.path.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2, default=str))

    def write(self, record):
        with self.path.open('a') as output:
            output.write(json.dumps(record, allow_nan=False) + '\n')


class PerformanceMonitor:
    def __init__(self, device, elapsed_seconds=0.0):
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.elapsed = elapsed_seconds

    def synchronize(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def begin(self, reset_peak=True):
        self.synchronize()
        if self.cuda and reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()

    def finish(self, samples, tokens, supervised_tokens, generated_tokens=0, extra_seconds=0.0):
        self.synchronize()
        seconds = time.perf_counter() - self.started + extra_seconds
        counts = torch.tensor([samples, tokens, supervised_tokens, generated_tokens],
                              dtype=torch.float64, device=self.device)
        # CPU runs deliberately report CUDA metrics as null, rather than zero usage.
        memory = ([torch.cuda.max_memory_allocated(self.device) / 2**20,
                   torch.cuda.max_memory_reserved(self.device) / 2**20] if self.cuda else [0., 0.])
        maxima = torch.tensor([seconds, *memory], dtype=torch.float64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        seconds, allocated, reserved = maxima.cpu().tolist()
        self.elapsed += seconds
        samples, tokens, supervised, generated = counts.cpu().tolist()
        return {'step_seconds': seconds, 'elapsed_seconds': self.elapsed,
                'samples': samples, 'input_tokens': tokens, 'supervised_tokens': supervised,
                'generated_tokens': generated,
                'samples_per_second': samples / seconds,
                'tokens_per_second': tokens / seconds,
                'supervised_tokens_per_second': supervised / seconds,
                'generated_tokens_per_second': generated / seconds,
                'cuda_peak_allocated_mib': allocated if self.cuda else None,
                'cuda_peak_reserved_mib': reserved if self.cuda else None}
