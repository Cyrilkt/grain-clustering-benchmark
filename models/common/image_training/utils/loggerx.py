"""Rank-zero JSON-lines logging for local runs."""
from pathlib import Path
import json
import torch
from .ops import is_root_worker

class LoggerX:

    def __init__(self, save_root, **kwargs):
        self.root = Path(save_root)
        self.enabled = is_root_worker()

    def msg(self, stats, step):
        if not self.enabled:
            return
        if not isinstance(stats, dict):
            stats = {f'value_{i}': value for i, value in enumerate(stats)}
        values = {key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value for key, value in stats.items()}
        values['step'] = int(step)
        with (self.root / 'training.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(values, allow_nan=False) + '\n')

    def msg_str(self, message):
        if self.enabled:
            print(message, flush=True)
