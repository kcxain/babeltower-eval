# BabelTower-Eval

This is an evaluation harness for the C-to-CUDA translation dataset of BabelTower. You can find detailed dataset in [HuggingFace](https://huggingface.co/datasets/kcxain/BabelTower).

## Installation

Ensure you are using Python 3.12 or later:
```bash
conda create -n babeltower python=3.12
conda activate babeltower
```

Install the repository:
```bash
git clone https://github.com/kcxain/babeltower-eval.git
cd babeltower-eval
pip install -r requirements.txt
```

To evaluate using a local model install vLLM
```bash
pip install vllm
```

## Usage
Our evaluation script is available at [eval_passk.py](https://github.com/kcxain/babeltower-eval/blob/master/babeltower_eval/eval_passk.py). For example, to evaluate `gpt-4o`:
```bash
model='gpt-4o-2024-05-13'
python -m babeltower_eval.eval_passk \
    --model_to_eval $model \
    --dataset 'kcxain/BabelTower' \
    --model_mode 'trans' \
    --sample_num 20 \
    --save_translated_code
```

### Parameter Details (evaluate.py)

| Parameter             | Type           | Description                                                                 |
|-----------------------|----------------|-----------------------------------------------------------------------------|
| `--model_to_eval`      | `str`          | Model name or local path (e.g., `gpt-3.5-turbo` or `path/to/local/model`).  |
| `--dataset`            | `str`          | HuggingFace dataset to evaluate (default: `kcxain/BabelTower`).             |
| `--file_path`          | `str`          | Path to a local translated code file (optional). Uses `model_to_eval` to generate if not provided. |
| `--model_mode`         | `PromptType`   | Type of prompt for translation (e.g., `trans` or `trans_trained`).          |
| `--sample_num`         | `int`          | Number of code samples to generate per input.                               |
| `--save_translated_code` | `flag`        | Flag to save the translated code to a file.                                 |

### Evaluation Output

The evaluation results include pass@1, pass@5, pass@10 for each case, along with the average values across the entire dataset.


## Citation

Please cite using the following bibtex entry:

```
@InProceedings{pmlr-v162-wen22b,
  title = 	 {{B}abel{T}ower: Learning to Auto-parallelized Program Translation},
  author =       {Wen, Yuanbo and Guo, Qi and Fu, Qiang and Li, Xiaqing and Xu, Jianxing and Tang, Yanlin and Zhao, Yongwei and Hu, Xing and Du, Zidong and Li, Ling and Wang, Chao and Zhou, Xuehai and Chen, Yunji},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  year = 	 {2022},
}
```