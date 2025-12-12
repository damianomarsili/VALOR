# ⭐ VALOR - No Labels, No Problem: Training Visual Reasoners with Multimodal Verifiers

This is the code for the paper [No Labels, No Problem: Training Visual Reasoners with Multimodal Verifiers](https://glab-caltech.github.io/valor/) by [Damiano Marsili](https://damianomarsili.github.io/) and [Georgia Gkioxari](https://georgiagkioxari.com/).

<div align="center">
  <a href="https://arxiv.org/abs/2512.08889"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://glab-caltech.github.io/valor/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
</div>

![image](docs/valor.png)

## 🚀 Quickstart
Clone the repo:
```bash
git clone https://github.com/damianomarsili/VALOR.git
```

We use `uv` to manage all dependencies. If your system does not have `uv`, install it via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Setup your environment:
```bash
cd VALOR
uv sync
```
⚠️ Note: This setup assumes CUDA 12.8 and Python 3.10.  If you are using a different CUDA version, you may need to install a version of `torch` and `flash-attn` compatible with your system.

VALOR uses [MoGe](https://github.com/microsoft/MoGe) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO). 

⚠️ Note: Prior to installing GroundingDINO, please follow the additional installation steps for setting the `CUDA_HOME` environment variable detailed on their [repo](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install).

Then, install them as follows:

```
uv run python -m pip install --no-build-isolation -e modules/GroundingDINO && \
uv run python -m pip install git+https://github.com/microsoft/MoGe.git
```

Additionally, VALOR uses GPT-5-mini for VQA. Please set your OpenAI API key to the following environment variable:
```bash
export OPENAI_API_KEY="API KEY"
```

All model checkpoints are hosted on Huggingface 🤗. We provide a script to download the trained GroundingDINO checkpoint. First, authenticate with `huggingface-cli`:
```bash
huggingface-cli login                
```
Then, download the checkpoint:
```bash
bash scripts/download_gd_checkpoint.sh
```

📓 For a brief exploration of VALOR's functionality, we have compiled a notebook `demo/quickstart.ipynb`. To run the notebook in the `uv` environment, run:
```bash
uv run jupyter lab
```
and navigate to the demo/quickstart.ipynb in the UI.

For full evaluations, please refer to the "Evaluating VALOR" section below.

## 📊 Evaluating VALOR
We use Huggingface 🤗 to load all datasets, except `GQA` and `TallyQA`, where we use the subsets provided in [GRIT](https://github.com/eric-ai-lab/GRIT#Setup).

To evaluate VALOR, please run the following code:
```bash
uv run python -m eval.eval --datasets omni3d-bench,tallyqa,realworldqa --out_dir eval_outputs/
```

## 🧠 Reasoning Training
VALOR uses LLM verifiers to improve the reasoning ability of an LLM. Training invokes Gemini and requires a Google Cloud account. We recommend authenticating via a [service account](https://developers.google.com/workspace/guides/create-credentials#service-account). Once authenticated, you should update the `GENAI_CREDS_PATH` and `GENAI_PROJECT_ID` variables in the `grounding_training/llm_training.sh` script.

Then, you can launch reasoning training via the following command:
```bash
uv run bash valor/reasoning_training/llm_training.sh
```
The data used to train VALOR is found in `reasoning_training/data/reasoning_data.jsonl`. 

After training, you can create a checkpoint compatible with `huggingface` by running the `merge` script in `verl`:
```bash
uv run python -m verl.model_merger merge --backend fsdp --local_dir /path/to/verl/checkpoint --target_dir /path/to/output/checkpoint
```

## 📌 Grounding Training
VALOR uses VLM verifiers to improve the visual grounding ability of a GroundingDINO model via automated hard-negative mining. Sourcing training data requires an OpenAI API key. Please set your OpenAI API key to the following environment variable:
```bash
export OPENAI_API_KEY="API KEY"
```
To generate training data, first download the pre-trained GroundingDINO model:
```bash
mkdir -p modules/GroundingDINO/weights/
wget -O modules/GroundingDINO/weights/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
Then, you can source training data via the following command:
```bash
uv run bash valor/grounding_training/source_training_data.sh
```

We build from the third-party [Open-GroundingDino repository](https://github.com/longzw1997/Open-GroundingDino) for training GroundingDINO 🦖. We thank the contributors of the repository for their efforts!

To launch training, you must first copy the list in `grounding_training/data/odvg/labels.txt` to the `label_list` entry at the bottom of the training config at `grounding_training/Open-GroundingDino/config/cfg_odvg.py`. Then, run the following command:
```bash
uv run bash valor/grounding_training/train_gd.sh
```
The trained checkpoint will default save to `valor/grounding_training/checkpoints/`. You can edit this target directory in the bash script `valor/grounding_training/train_gd.sh`.

## 📚 Citation
If you use VALOR in your research, please consider citing our work:
```bibtex
@misc{marsili2025labelsproblemtrainingvisual,
      title={No Labels, No Problem: Training Visual Reasoners with Multimodal Verifiers}, 
      author={Damiano Marsili and Georgia Gkioxari},
      year={2025},
      eprint={2512.08889},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.08889}, 
}
```
