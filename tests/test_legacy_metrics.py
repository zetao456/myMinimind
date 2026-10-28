"""Check that the original pretraining and Full SFT CLIs produce usable metrics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LegacyMetricTests(unittest.TestCase):
    def test_original_training_entry_points(self):
        for stage in ('pretrain', 'full_sft'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                work = directory / 'work'
                work.mkdir()
                data = directory / 'data.jsonl'
                records = ([{'text': 'Small language model test.'}] if stage == 'pretrain' else
                           [{'conversations': [{'role': 'user', 'content': '你好'},
                                               {'role': 'assistant', 'content': '你好！'}]}])
                data.write_text(''.join(json.dumps(row) + '\n' for row in records))
                metrics = directory / 'metrics.jsonl'
                command = [sys.executable, str(ROOT / f'trainer/train_{stage}.py'),
                    '--from_weight', 'none', '--hidden_size', '32', '--num_hidden_layers', '1',
                    '--data_path', str(data), '--device', 'cpu', '--epochs', '1', '--batch_size', '1',
                    '--accumulation_steps', '1', '--num_workers', '0', '--max_seq_len', '96',
                    '--save_dir', str(directory / 'out'), '--metrics_path', str(metrics)]
                # Original tokenizer_path is relative to the working directory.
                (directory / 'model').symlink_to(ROOT / 'model', target_is_directory=True)
                result = subprocess.run(command, cwd=work, capture_output=True, text=True,
                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                         'HF_DATASETS_CACHE': str(directory / 'hf_cache')}, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                row = json.loads(metrics.read_text().splitlines()[0])
                self.assertGreater(row['input_tokens'], 0)
                self.assertGreater(row['supervised_tokens'], 0)
                self.assertGreater(row['tokens_per_second'], 0)
                self.assertIsNone(row['cuda_peak_allocated_mib'])
                self.assertIn('learning_rate', row)
                self.assertIn('loss', row)


if __name__ == '__main__':
    unittest.main()
