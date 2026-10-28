import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import AutoTokenizer
from dataset.alignment_dataset import DPODataset
from model.model_myminimind import MiniMindForCausalLM, MyMiniMindConfig
from model.model_lora import (apply_lora, mark_only_lora_trainable, save_lora,
                              load_lora, merge_lora, LoRALinear)
from trainer.alignment_objectives import (sequence_logps, dpo_loss, completion_mask,
    group_advantages, grpo_loss, exact_match_reward)

ROOT = Path(__file__).resolve().parents[1]


def tiny_model():
    return MiniMindForCausalLM(MyMiniMindConfig(hidden_size=32, num_hidden_layers=1,
        vocab_size=64, max_position_embeddings=128))


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)

    def setUp(self):
        torch.manual_seed(7)

    def test_lora_freeze_roundtrip_and_merge(self):
        base = tiny_model().eval()
        original = copy.deepcopy(base.state_dict())
        x = torch.randint(3, 64, (2, 10))
        before = base(x).logits.detach()
        apply_lora(base, rank=3, alpha=7, target_modules='q_proj,k_proj,v_proj,o_proj,up_proj')
        torch.testing.assert_close(before, base(x).logits)
        mark_only_lora_trainable(base)
        optimizer = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=0.02)
        base(x, labels=x).loss.backward()
        optimizer.step()
        for k, value in original.items():
            torch.testing.assert_close(base.state_dict()[k], value, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(v, torch.zeros_like(v)) for k, v in base.state_dict().items() if '.lora.B.' in k))
        with tempfile.TemporaryDirectory() as temp:
            adapter, merged = Path(temp) / 'adapter.pth', Path(temp) / 'merged.pth'
            save_lora(base, adapter)
            restored = tiny_model().eval()
            restored.load_state_dict(original)
            load_lora(restored, adapter)
            torch.testing.assert_close(base(x).logits, restored(x).logits)
            self.assertEqual(restored.model.layers[0].self_attn.k_proj.lora.rank, 3)
            merge_lora(restored, adapter, merged)
            plain = tiny_model().eval()
            plain.load_state_dict(torch.load(merged, weights_only=True), strict=True)
            torch.testing.assert_close(base(x).logits, plain(x).logits, atol=2e-6, rtol=2e-5)
            legacy = Path(temp) / 'legacy.pth'
            # Legacy adapters have scale 1 and no metadata.
            legacy_base = tiny_model().eval()
            apply_lora(legacy_base)
            torch.save({k: v for k, v in legacy_base.state_dict().items() if '.lora.' in k}, legacy)
            load_lora(tiny_model(), legacy)

    def test_lora_rejects_duplicate_injection(self):
        model = tiny_model()
        apply_lora(model)
        with self.assertRaises(ValueError):
            apply_lora(model)

    def test_dpo_data_masks_history_and_padding(self):
        ds = DPODataset(ROOT / 'dataset/examples/dpo.jsonl', self.tokenizer, 128, 64)
        for row in ds:
            for prefix in ('chosen', 'rejected'):
                labels = row[prefix + '_labels']
                mask = row[prefix + '_attention_mask']
                self.assertTrue((labels[mask == 0] == -100).all())
                text = self.tokenizer.decode(labels[labels != -100])
                self.assertIn(self.tokenizer.eos_token, text)
                self.assertNotIn('<|im_start|>user', text)
                self.assertEqual(labels[0].item(), -100)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.jsonl'
            path.write_text(json.dumps({'prompt': 'hi', 'chosen': 'same', 'rejected': 'same'}) + '\n')
            with self.assertRaises(ValueError):
                DPODataset(path, self.tokenizer, 64, 32)[0]
            path.write_text(json.dumps({
                'chosen': [{'role': 'user', 'content': '1+1?'},
                           {'role': 'assistant', 'content': '2', 'reasoning_content': 'add one'}],
                'rejected': [{'role': 'user', 'content': '1+1?'},
                             {'role': 'assistant', 'content': '3', 'reasoning_content': 'guess'}]}) + '\n')
            row = DPODataset(path, self.tokenizer, 128, 64)[0]
            labels = row['chosen_labels']
            self.assertIn('add one', self.tokenizer.decode(labels[labels != -100]))

    def test_sequence_logps_matches_manual_sum(self):
        model = tiny_model().eval()
        ids = torch.tensor([[1, 4, 5, 6, 0]])
        labels = torch.tensor([[-100, -100, 5, 6, -100]])
        logits = model(ids).logits[:, :-1].log_softmax(-1)
        expected = logits[0, 1, 5] + logits[0, 2, 6]
        actual = sequence_logps(model, ids, labels)[0]
        torch.testing.assert_close(actual, expected)

    def test_dpo_gradient_prefers_chosen(self):
        model = tiny_model().eval()
        ref = copy.deepcopy(model).requires_grad_(False)
        ids = torch.tensor([[1, 3, 7, 2], [1, 3, 9, 2]])
        labels = ids.clone()
        labels[:, :2] = -100
        with torch.no_grad():
            ref_c, ref_r = sequence_logps(ref, ids, labels).chunk(2)
        chosen, rejected = sequence_logps(model, ids, labels).chunk(2)
        loss, _ = dpo_loss(chosen, rejected, ref_c, ref_r)
        self.assertAlmostEqual(loss.item(), 0.693147, places=5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss.backward()
        optimizer.step()
        c, r = sequence_logps(model, ids, labels).chunk(2)
        self.assertGreater(((c-r) - (chosen-rejected)).item(), 0)
        self.assertTrue(all(p.grad is None for p in ref.parameters()))

    def test_grpo_eos_and_group_normalization(self):
        mask = completion_mask(torch.tensor([[4, 2, 2, 0], [4, 5, 6, 7], [2, 0, 0, 0]]), 2)
        self.assertEqual(mask.tolist(), [[True, True, False, False], [True]*4, [True, False, False, False]])
        advantages = group_advantages(torch.tensor([1., 0., 5., 5.]), 2)
        self.assertGreater(advantages[0], 0)
        self.assertLess(advantages[1], 0)
        torch.testing.assert_close(advantages[2:], torch.zeros(2))

    def test_grpo_gradient_clip_and_kl(self):
        old = torch.tensor([[-2., -3.], [-2., -3.]])
        logps = old.clone().requires_grad_()
        mask = torch.tensor([[True, False], [True, True]])
        advantages = torch.tensor([1., -1.])
        loss, metrics = grpo_loss(logps, old, old, mask, advantages)
        self.assertAlmostEqual(metrics['kl'].item(), 0)
        loss.backward()
        self.assertLess(logps.grad[0, 0], 0)
        self.assertEqual(logps.grad[0, 1], 0)
        self.assertGreater(logps.grad[1, 0], 0)
        shifted = (old + torch.tensor([[1., 0.], [-1., -1.]])).requires_grad_()
        clipped, metrics = grpo_loss(shifted, old, old, mask, advantages, beta=0)
        clipped.backward()
        torch.testing.assert_close(shifted.grad, torch.zeros_like(shifted))
        self.assertGreater(metrics['kl'], 0)
        self.assertEqual(metrics['clip_fraction'], 1)

    def test_exact_match_reward(self):
        texts = ['work\n<answer> ２ </answer>', 'work #### 4', r'work \boxed{5}', '12']
        rows = [{'answer': '2'}, {'answer': ['4', '4.0']}, {'answer': '5'}, {'answer': '2'}]
        self.assertEqual(exact_match_reward([], texts, rows), [1., 1., 1., 0.])
        with self.assertRaises(ValueError):
            exact_match_reward([], ['hi'], [{}])


if __name__ == '__main__':
    unittest.main()
