import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trainer.metrics import MetricWriter, PerformanceMonitor
from experiments.plot_metrics import plot_metrics


class MetricsTests(unittest.TestCase):
    def test_resume_discards_records_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run.metrics.jsonl'
            writer = MetricWriter(path, {'step_unit': 'optimizer_update'})
            for step in range(1, 5):
                writer.write({'step': step, 'elapsed_seconds': step * 2.})
            writer = MetricWriter(path, {}, resume_step=2)
            self.assertEqual(writer.elapsed_seconds, 4.)
            writer.write({'step': 3, 'elapsed_seconds': 6.})
            self.assertEqual([json.loads(line)['step'] for line in path.read_text().splitlines()], [1, 2, 3])

    def test_cpu_throughput_and_unknown_cuda_memory(self):
        monitor = PerformanceMonitor('cpu')
        with patch('trainer.metrics.time.perf_counter', side_effect=[10., 12.]):
            monitor.begin()
            result = monitor.finish(4, 100, 20)
        self.assertEqual(result['tokens_per_second'], 50.)
        self.assertEqual(result['supervised_tokens_per_second'], 10.)
        self.assertEqual(result['step_seconds'], 2.)
        self.assertIsNone(result['cuda_peak_allocated_mib'])
        self.assertIsNone(result['cuda_peak_reserved_mib'])


    def test_plot_exports_and_missing_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run.metrics.jsonl'
            writer = MetricWriter(path, {})
            writer.write({'step': 1, 'loss': .5, 'learning_rate': .001, 'tokens_per_second': 40,
                          'cuda_peak_allocated_mib': None})
            result = plot_metrics([path], Path(temp) / 'plots')
            self.assertGreater(result.stat().st_size, 1000)
            self.assertTrue(result.with_suffix('.svg').exists())
            self.assertTrue(result.with_suffix('.pdf').exists())


if __name__ == '__main__':
    unittest.main()
