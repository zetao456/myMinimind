"""Train with outcome-supervised Group Relative Policy Optimization."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.train_alignment import main

if __name__ == '__main__':
    main('grpo')
