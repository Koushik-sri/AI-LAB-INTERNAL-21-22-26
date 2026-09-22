import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fine_tune_pipeline import FineTunePipeline

@pytest.fixture
def data_file(tmp_path):
    p = tmp_path / "test_domain.jsonl"
    p.write_text(json.dumps({"instruction": "Q", "response": "A"}) + "\n", encoding="utf-8")
    return str(p)

def test_fine_tune_pipeline(data_file):
    pipe = FineTunePipeline(data_file)
    assert len(pipe.dataset) == 1
    train_res = pipe.train_lora_adapter(num_epochs=2)
    assert train_res["status"] == "COMPLETED"
    assert len(train_res["training_history"]) == 2
    eval_res = pipe.evaluate_base_vs_adapted()
    assert "fine_tuned_lora_model" in eval_res
