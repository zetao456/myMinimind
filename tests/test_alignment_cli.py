"""CPU end-to-end checks: real entry points, artifacts, inference, exact resume."""
import os
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import AutoTokenizer
from model.model_lora import load_lora
from model.model_myminimind import MiniMindForCausalLM, MyMiniMindConfig

ROOT = Path(__file__).resolve().parents[1]


class AlignmentCLITests(unittest.TestCase):
    def run_training(self, stage, output, steps, resume=False, extra=(), distributed=False):
        command = [sys.executable, str(ROOT / f'trainer/train_{stage}.py'),
            '--data_path', str(ROOT / f'dataset/examples/{stage}.jsonl'),
            '--from_weight', 'none', '--hidden_size', '32', '--num_hidden_layers', '1',
            '--device', 'cpu', '--dtype', 'float32', '--epochs', '2', '--batch_size', '1',
            '--accumulation_steps', '2', '--max_seq_len', '96', '--max_steps', str(steps),
            '--save_dir', str(output), '--checkpoint_dir', str(output), '--log_interval', '1',
            '--learning_rate', '0.001']
        if stage != 'lora':
            command += ['--max_prompt_len', '64']
        if stage == 'grpo':
            command += ['--max_new_tokens', '4', '--group_size', '2', '--num_iterations', '2',
                        '--reward_fn', 'tests.grpo_reward_fixture:reward']
        if resume:
            command += ['--from_resume', '1']
        command += list(extra)
        if distributed:
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc_per_node=2', *command[1:]]
        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
               'PYTHONPATH': str(ROOT), 'HF_DATASETS_CACHE': str(output / 'hf_cache')}
        result = subprocess.run(command, cwd='/tmp', env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        prefix = extra[extra.index('--save_weight') + 1] if '--save_weight' in extra else ('lora_sft' if stage == 'lora' else stage)
        suffix = '_moe' if '--use_moe' in extra and extra[extra.index('--use_moe') + 1] == '1' else ''
        return torch.load(output / f'{prefix}_32{suffix}_resume.pth',
                          map_location='cpu', weights_only=False)

    def test_adapter_alignment_and_moe(self):
        for stage in ('dpo', 'grpo'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp:
                result = self.run_training(stage, Path(temp), 2 if stage == 'grpo' else 1,
                                           extra=['--use_lora'])
                self.assertTrue(any('.lora.' in k for k in result['model']))
        with tempfile.TemporaryDirectory() as temp:
            # MoE exercises variable expert usage with adapters and reference scoring.
            path = Path(temp)
            self.run_training('grpo', path, 2, extra=['--use_lora', '--use_moe', '1',
                                                    '--save_weight', 'grpo_moe_test'])

    def test_two_rank_cpu_dpo(self):
        try:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
        except PermissionError:
            self.skipTest('DDP needs local sockets; this sandbox disables them')
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_training('dpo', Path(temp), 1, distributed=True)
            self.assertEqual(result['world_size'], 2)
            self.assertEqual(len(result['rng']), 2)

    def test_train_save_resume_and_generate(self):
        torch.set_num_threads(2)
        for stage in ('lora', 'dpo', 'grpo'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp:
                uninterrupted, resumed = Path(temp) / 'full', Path(temp) / 'resumed'
                # GRPO does two optimizer passes per rollout; stop at complete rollouts.
                stop, end = (2, 4) if stage == 'grpo' else (1, 3)
                full = self.run_training(stage, uninterrupted, end)
                first = self.run_training(stage, resumed, stop)
                final = self.run_training(stage, resumed, end, resume=True)
                self.assertEqual(final['update'], end)
                self.assertTrue(any(not torch.equal(v, first['model'][k])
                                    for k, v in final['model'].items()))
                for key in full['model']:
                    torch.testing.assert_close(full['model'][key], final['model'][key], rtol=0, atol=0,
                                               msg=f'{stage}: resume differs at {key}')
                if final['reference'] is not None:
                    for key, value in final['reference'].items():
                        torch.testing.assert_close(value, first['reference'][key], rtol=0, atol=0)
                tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model')
                model = MiniMindForCausalLM(MyMiniMindConfig(hidden_size=32, num_hidden_layers=1,
                                                            vocab_size=len(tokenizer))).eval()
                path = resumed / f'{"lora_sft" if stage == "lora" else stage}_32.pth'
                if stage == 'lora':
                    # Base parameters are held frozen throughout adapter training.
                    model.load_state_dict({k: v for k, v in final['model'].items() if '.lora.' not in k})
                    load_lora(model, path)
                else:
                    model.load_state_dict(torch.load(path, weights_only=True), strict=True)
                generated = model.generate(torch.tensor([[1, 4, 5]]), max_new_tokens=2,
                                           eos_token_id=None, do_sample=False)
                self.assertEqual(generated.shape, (1, 5))


if __name__ == '__main__':
    unittest.main()
