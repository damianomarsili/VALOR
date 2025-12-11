# ⭐ VALOR: No Labels, No Problem: Training Visual Reasoners with Multimodal Verifiers

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
⚠️ Note: This setup assumes CUDA 12.8 and Python 3.10.  If you are using a different CUDA version, you may need to install a version of `torch` compatible with your CUDA version.

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
Coming soon!

## 📌 Grounding Training
Coming soon!

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
