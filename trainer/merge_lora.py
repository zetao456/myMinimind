"""Merge an adapter into a plain MiniMind checkpoint for eval_llm.py."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.model_myminimind import MyMiniMindConfig, MiniMindForCausalLM
from model.model_lora import merge_lora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_path', required=True)
    parser.add_argument('--lora_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--use_moe', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    if Path(args.output_path).resolve() in (Path(args.base_path).resolve(), Path(args.lora_path).resolve()):
        parser.error('output_path must differ from the input checkpoints')
    model = MiniMindForCausalLM(MyMiniMindConfig(hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
    model.load_state_dict(torch.load(args.base_path, map_location='cpu', weights_only=True), strict=True)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    merge_lora(model, args.lora_path, args.output_path)
    print(f'Merged model saved to {args.output_path}')


if __name__ == '__main__':
    main()
