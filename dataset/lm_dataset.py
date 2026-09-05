from torch.utils.data import Dataset
import torch
import json
import os
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Features, Value
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
    
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# === 预缓存：fork workers 直接写 int16 memmap，磁盘只存一份 ===

_all_conversations = None  # 父进程设好，fork 后 worker 通过 COW 共享读取


class CachedSFTDataset(Dataset):
    """加载预缓存的 tokenized 数据，int16 memmap -> int64 tensor"""
    def __init__(self, cache_dir):
        super().__init__()
        shape = tuple(int(x) for x in np.load(os.path.join(cache_dir, 'shape.npy')))
        self.input_ids = np.memmap(os.path.join(cache_dir, 'input_ids.dat'), dtype=np.int16, mode='r', shape=shape)
        self.labels = np.memmap(os.path.join(cache_dir, 'labels.dat'), dtype=np.int16, mode='r', shape=shape)
        self.max_length = shape[1]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return (torch.from_numpy(self.input_ids[index].copy().astype(np.int64)),
                torch.from_numpy(self.labels[index].copy().astype(np.int64)))


def _tokenize_chunk_memmap(args):
    """worker 进程：从 _all_conversations (COW) 读原始文本，tokenize 后直写 memmap"""
    start, end, cache_dir, max_length, bos_assistant_ids, eos_nl_ids, pad_token_id = args

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(cache_dir, '_tok'))

    shape = tuple(int(x) for x in np.load(os.path.join(cache_dir, 'shape.npy')))
    input_ids_mmap = np.memmap(os.path.join(cache_dir, 'input_ids.dat'), dtype=np.int16, mode='r+', shape=shape)
    labels_mmap = np.memmap(os.path.join(cache_dir, 'labels.dat'), dtype=np.int16, mode='r+', shape=shape)

    for idx in range(start, end):
        conversations = pre_processing_chat(_all_conversations[idx])
        messages, tools = [], None
        for m in conversations:
            m = dict(m)
            if m.get("role") == "system" and m.get("tools"):
                tools = json.loads(m["tools"]) if isinstance(m["tools"], str) else m["tools"]
            if m.get("tool_calls") and isinstance(m["tool_calls"], str):
                m["tool_calls"] = json.loads(m["tool_calls"])
            messages.append(m)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
        prompt = post_processing_chat(prompt)
        ids = tokenizer(prompt).input_ids[:max_length]
        ids += [pad_token_id] * (max_length - len(ids))

        labels = [-100] * len(ids)
        pos = 0
        while pos < len(ids):
            if ids[pos:pos + len(bos_assistant_ids)] == bos_assistant_ids:
                s = pos + len(bos_assistant_ids)
                e = s
                while e < len(ids):
                    if ids[e:e + len(eos_nl_ids)] == eos_nl_ids:
                        break
                    e += 1
                for j in range(s, min(e + len(eos_nl_ids), max_length)):
                    labels[j] = ids[j]
                pos = e + len(eos_nl_ids) if e < len(ids) else len(ids)
            else:
                pos += 1

        input_ids_mmap[idx] = np.array(ids, dtype=np.int16)
        labels_mmap[idx] = np.array(labels, dtype=np.int16)

    input_ids_mmap.flush()
    labels_mmap.flush()


def build_sft_cache(data_path, tokenizer, max_length, cache_dir, num_proc=None):
    """多进程并行 tokenize，int16 memmap 落盘，磁盘只存一份"""
    import multiprocessing as mp

    os.makedirs(cache_dir, exist_ok=True)
    if num_proc is None:
        num_proc = max(1, os.cpu_count() - 1)

    features = Features({'conversations': [{
        'role': Value('string'), 'content': Value('string'),
        'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')
    }]})
    raw_ds = load_dataset('json', data_files=data_path, split='train', features=features)
    n = len(raw_ds)

    # 提到 Python list，fork 后 worker 通过全局变量 COW 共享
    global _all_conversations
    _all_conversations = list(raw_ds['conversations'])
    del raw_ds

    # tokenizer 存盘，worker 各自加载
    tokenizer.save_pretrained(os.path.join(cache_dir, '_tok'))

    bos_assistant_ids = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
    eos_nl_ids = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    # int16 memmap（vocab 6400 + -100 都在 int16 范围），磁盘 ~2.8G
    shape = (n, max_length)
    input_ids_mmap = np.memmap(os.path.join(cache_dir, 'input_ids.dat'), dtype=np.int16, mode='w+', shape=shape)
    labels_mmap = np.memmap(os.path.join(cache_dir, 'labels.dat'), dtype=np.int16, mode='w+', shape=shape)
    np.save(os.path.join(cache_dir, 'shape.npy'), np.array(shape))
    del input_ids_mmap, labels_mmap

    chunk_size = max(1, (n + num_proc - 1) // num_proc)
    tasks = [(i, min(i + chunk_size, n), cache_dir, max_length,
              bos_assistant_ids, eos_nl_ids, tokenizer.pad_token_id)
             for i in range(0, n, chunk_size)]

    with mp.Pool(num_proc) as pool:
        list(tqdm(pool.imap_unordered(_tokenize_chunk_memmap, tasks),
                  total=len(tasks), desc=f"Pre-tokenizing ({num_proc} procs)"))

    _all_conversations = None
    print(f"Cached {n} samples to {cache_dir} (int16 memmap)")