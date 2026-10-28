# Simulate Before You Act

Code and data for the paper *"Simulate Before You Act: A Robust Digital-Twin Safety Gate for LLM and Rule-Based Supervisors in Process Fault Handling"* by I. Dhinakaran, A. Nair and S.P. Agrawal (VIT).

## Requirements

- Python 3.11+ (tested with 3.14)
- [Ollama](https://ollama.com) with the models `qwen2.5:3b` and `qwen2.5:7b`

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b && ollama pull qwen2.5:7b
```

Before running the LLM experiments, start Ollama in a second terminal with `ollama serve`.

## Reproducing the paper

| Paper | Command | Output |
|---|---|---|
| Experiment 1 (oracle twin) | `python experiment_v2.py` | results_v2.csv |
| Experiment 2 (realistic twins, unseen faults) | `python experiment_v3.py` | results_v3.csv |
| Experiment 3 (gate study) | `python gate_study.py`, then `python gate_margin.py` | gate_summary.csv, gate_by_fault.csv, gate_margin.csv |

`llm_cache.json` stores every LLM answer by prompt hash, so the experiments reproduce the paper's numbers exactly without calling the models again. Delete it to query the models afresh.

Seeds 0–19 are used throughout.
