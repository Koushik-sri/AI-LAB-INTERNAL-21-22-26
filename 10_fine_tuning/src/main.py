"""Main entry point for Fine-Tuning for Domain Adaptation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fine_tune_pipeline import FineTunePipeline

def main():
    print("=" * 60)
    print("    FINE-TUNING FOR DOMAIN ADAPTATION - PROJECT 10")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "domain_data.jsonl")

    pipeline = FineTunePipeline(data_file)
    print(f"[*] Loaded {len(pipeline.dataset)} domain fine-tuning examples.")

    print("\n[*] Initializing LoRA Training Loop...")
    train_res = pipeline.train_lora_adapter(num_epochs=3)
    print("    LoRA Config:", train_res["lora_config"])
    print("    Training Loss History:")
    for h in train_res["training_history"]:
        print(f"      Epoch {h['epoch']}: Loss = {h['loss']}")

    print("\n[*] Evaluating Base Model vs. LoRA-Adapted Model...")
    eval_res = pipeline.evaluate_base_vs_adapted()
    print("    Base Model Metrics:", eval_res["base_model"])
    print("    Adapted Model Metrics:", eval_res["fine_tuned_lora_model"])
    print("    Summary Delta:", eval_res["improvement"])

    print("\n[OK] Project 10 Fine-Tuning execution finished successfully.")

if __name__ == "__main__":
    main()
