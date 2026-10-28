from pathlib import Path
from types import SimpleNamespace
import subprocess
import unittest

import torch
from transformers import AutoTokenizer
from dataset.alignment_dataset import DPODataset, PreferenceSFTDataset
from model.model_myminimind import MiniMindForCausalLM, MyMiniMindConfig
from trainer.rollout import sample_completions

ROOT = Path(__file__).resolve().parents[1]


class PreferenceExperimentTests(unittest.TestCase):
    def test_sft_and_dpo_share_chosen_tokens(self):
        tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
        pairs = DPODataset(ROOT / 'dataset/examples/dpo.jsonl', tokenizer, 128, 64)
        sft = PreferenceSFTDataset(ROOT / 'dataset/examples/dpo.jsonl', tokenizer, 128, 64)
        for index in range(len(sft)):
            ids, labels = sft[index]
            torch.testing.assert_close(ids, pairs[index]['chosen_input_ids'], rtol=0, atol=0)
            torch.testing.assert_close(labels, pairs[index]['chosen_labels'], rtol=0, atol=0)

    def test_batched_left_padding_matches_individual_greedy(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = MiniMindForCausalLM(MyMiniMindConfig(hidden_size=32, num_hidden_layers=1,
            vocab_size=64, max_position_embeddings=64)).eval()
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        prompts = [[1, 4, 5], [1, 8, 9, 10, 11, 12]]
        batched = sample_completions(model, tokenizer, prompts, 'cpu', 4, group_size=2, do_sample=False)
        individual = [sample_completions(model, tokenizer, [prompt], 'cpu', 4, do_sample=False)[0]
                      for prompt in prompts]
        for i, completion in enumerate(batched):
            torch.testing.assert_close(completion, individual[i // 2], rtol=0, atol=0)

    def test_generated_metrics_are_git_ignored(self):
        paths = ['out/example.metrics.jsonl', 'example.metrics.jsonl',
                 'example.metrics.meta.json', 'experiments/runs/example/model.metrics.jsonl',
                 'dataset/prepared/zh_preference/train.jsonl']
        result = subprocess.run(['git', 'check-ignore', '--stdin'], input='\n'.join(paths),
                                text=True, capture_output=True, cwd=ROOT, check=True)
        self.assertEqual(set(result.stdout.splitlines()), set(paths))


if __name__ == '__main__':
    unittest.main()
