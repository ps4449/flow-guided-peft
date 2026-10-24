"""ScienceQA entry point using the verified AdaLoRA engine with shared partial-batch handling."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/adalora_reproduction'))
import adalora_train_engine as train_lora
if __name__=='__main__':train_lora.main()
