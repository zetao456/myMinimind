"""JSONL preference pairs and verifiable-reward prompts."""
import json
import torch
from torch.utils.data import Dataset


class JsonlDataset(Dataset):
    def __init__(self, path):
        self.path, self.offsets = str(path), []
        with open(path, 'rb') as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f'Empty JSONL dataset: {path}')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with open(self.path, 'rb') as source:
            source.seek(self.offsets[index])
            return json.loads(source.readline())


def prompt_messages(prompt):
    messages = [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt
    if not isinstance(messages, list) or not messages or messages[-1].get('role') not in ('user', 'tool'):
        raise ValueError('prompt must be a string or chat messages ending in user/tool')
    return messages


def encode_prompt(tokenizer, messages, max_length, open_thinking=False):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         open_thinking=open_thinking)
    return tokenizer(text, add_special_tokens=False).input_ids[-max_length:]


class DPODataset(JsonlDataset):
    def __init__(self, path, tokenizer, max_length=768, max_prompt_length=384):
        super().__init__(path)
        if not 1 <= max_prompt_length <= max_length - 2:
            raise ValueError('max_prompt_length must be between 1 and max_length - 2')
        self.tokenizer, self.max_length = tokenizer, max_length
        self.max_prompt_length = max_prompt_length

    def __getitem__(self, index):
        row = super().__getitem__(index)
        chosen, rejected = row['chosen'], row['rejected']
        if isinstance(chosen, list) and isinstance(rejected, list):
            if (not chosen or not rejected or chosen[:-1] != rejected[:-1]
                    or chosen[-1]['role'] != 'assistant' or rejected[-1]['role'] != 'assistant'):
                raise ValueError('chosen/rejected must share history and end with assistant')
            messages = prompt_messages(chosen[:-1])
            chosen, rejected = chosen[-1], rejected[-1]
        else:
            messages = prompt_messages(row['prompt'])
            chosen = {'role': 'assistant', 'content': chosen}
            rejected = {'role': 'assistant', 'content': rejected}
        prefix = self.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, open_thinking=True)
        prompt_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids[-self.max_prompt_length:]
        result = {}
        for name, response in [('chosen', chosen), ('rejected', rejected)]:
            text = self.tokenizer.apply_chat_template(messages + [response], tokenize=False,
                                                       open_thinking=False)
            if not text.startswith(prefix):
                raise ValueError('Chat template does not preserve the prompt prefix')
            completion = self.tokenizer(text[len(prefix):], add_special_tokens=False).input_ids
            completion = completion[:self.max_length - len(prompt_ids)]
            if not completion:
                raise ValueError('Preference response has no tokens')
            ids = prompt_ids + completion
            pad = self.max_length - len(ids)
            result[name + '_input_ids'] = torch.tensor(ids + [self.tokenizer.pad_token_id] * pad)
            result[name + '_labels'] = torch.tensor([-100] * len(prompt_ids) + completion + [-100] * pad)
            result[name + '_attention_mask'] = torch.tensor([1] * len(ids) + [0] * pad)
        if torch.equal(result['chosen_input_ids'], result['rejected_input_ids']):
            raise ValueError('chosen/rejected are identical after truncation; increase max_seq_len')
        return result


class GRPODataset(JsonlDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        row['messages'] = prompt_messages(row.get('prompt', row.get('messages')))
        return row


def collate_prompts(rows):
    return rows


class PreferenceSFTDataset(DPODataset):
    """Use exactly the chosen tokens/masks from a real preference pair, without augmentation."""
    def __getitem__(self, index):
        row = super().__getitem__(index)
        return row['chosen_input_ids'], row['chosen_labels']
