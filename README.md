<div align="center">
  <img src="assets/index_icon.png" width="220" alt="IndexTTS2 icon" />
</div>

<div align="center">
  <a href="docs/README_zh.md" style="font-size: 24px">简体中文</a>
</div>

# IndexTTS2

IndexTTS2 is the latest generation of the Index Team's zero-shot text-to-speech (TTS) system. The model couples high-fidelity voice cloning with expressive emotional rendering and introduces a controllable-duration autoregressive decoding strategy so you can lock narration to exact timelines or let it flow naturally.

## Highlights
- **Precise or free-form timing** – switch between fixed-token generation for frame-accurate dubs and unconstrained autoregressive speech.
- **Disentangled style control** – drive the speaker identity with a voice prompt while steering emotion from a separate style clip, vector, or textual description.
- **Fast, hardware-aware inference** – optional FP16, DeepSpeed acceleration, and fused CUDA kernels keep latency low on modern GPUs while still falling back to CPU-only runs.
- **Ready-to-use demos** – launch a Gradio-powered web UI, call from the command line, or embed `IndexTTS2` directly into Python pipelines.

## Quick Links
- [Paper (arXiv 2506.21619)](https://arxiv.org/abs/2506.21619)
- [Project page](https://index-tts.github.io/index-tts2.github.io/)
- [Hugging Face demo](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo)
- [Model download (Hugging Face)](https://huggingface.co/IndexTeam/IndexTTS-2)
- [Model download (ModelScope)](https://modelscope.cn/models/IndexTeam/IndexTTS-2)

> ℹ️ Looking for the original Simplified Chinese documentation? See [`docs/README_zh.md`](docs/README_zh.md).

## Getting started

### 1. Install prerequisites
- [git](https://git-scm.com/downloads) and [git-lfs](https://git-lfs.com/)
- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/) (recommended to guarantee dependency resolution)
- Python 3.10 or newer and an NVIDIA GPU with CUDA 12.8+ for accelerated inference (CPU execution is supported but significantly slower)

### 2. Clone the repository and fetch LFS assets
If you have not already cloned the project, do so once and change into the repository directory. All subsequent commands assume
they are executed from the root of the cloned repository.

```bash
git clone https://github.com/index-tts/index-tts2.git
cd index-tts2
git lfs install
git lfs pull
```

### 3. Create the environment and install dependencies
`uv sync` builds a virtual environment under `.venv` and installs all optional extras used by the web UI, DeepSpeed, and tooling.

```bash
uv sync --all-extras
```

If you only need a subset of functionality, replace `--all-extras` with the extras you require (for example, `--extra webui` or `--extra deepspeed`). Mirrors can be supplied via `--default-index` when necessary.

### 4. Download the pretrained checkpoints
Use either Hugging Face or ModelScope to populate the `checkpoints/` directory before inference:

```bash
uv tool install "huggingface_hub[cli]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```

or

```bash
uv tool install modelscope
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

You can optionally set `HF_ENDPOINT=https://hf-mirror.com` to speed up downloads from mirrored endpoints.

### 5. Verify GPU availability (optional)
```bash
uv run tools/gpu_check.py
```

## Usage

### Launch the web UI
```bash
uv run python webui.py --model_dir checkpoints
```

By default the interface starts on <http://127.0.0.1:7860>. Run `uv run python webui.py -h` to discover flags for FP16 decoding, DeepSpeed acceleration, custom ports, and more. MP3 export is enabled automatically when the necessary encoder is available.

#### Extras tab tools
The **Extras** tab bundles utility workflows that complement the main TTS interface:

- **Isolate Vocals (Deep Extract)** – mirrors the Demucs two-stem run used by ComfyUI DeepExtract V2 to split a mixed track into vocal and instrumental stems.
- **Merge MP3 Files** – concatenate multiple MP3s and optionally keep or rewrite their chapter markers.
- **MP3 Chapter Editor** – inspect and rename embedded chapter metadata before saving back to disk.

The Extras workflows depend on optional packages that are not required for core synthesis. Install only the pieces you plan to use:

| Feature | Required packages | Installation command |
| --- | --- | --- |
| Vocal isolation (Demucs) | `demucs`, `pydub`, `mutagen` | `pip install demucs pydub mutagen`<br/>or `uv sync --extra vocal_isolation` |
| MP3 chapter editing | `pydub`, `mutagen` | `pip install pydub mutagen` |
| VoiceForge OCR helpers | `deepseek-ocr` | `pip install deepseek-ocr` |

If you encounter warnings about a missing module (for example `The 'demucs' package is required for vocal isolation` or `The 'deepseek-ocr' package is not installed`), install the listed packages inside the same Python environment where you run `python webui.py` or the VoiceForge tooling. Cloning the repository is sufficient—you do **not** need to fetch any additional projects beyond installing the required Python packages. Running `pip install deepseek-ocr` downloads the published wheel from PyPI, so there is no separate Git checkout or manual requirements step needed for the OCR helper.

### Command line synthesis
Use the packaged CLI to synthesize speech directly from the terminal. The command below clones the voice in `examples/voice_01.wav` and saves the result to `gen.wav` (overwriting the file when `--force` is present).

```bash
uv run python -m indextts.cli "Hello from IndexTTS2!" \
  --voice examples/voice_01.wav \
  --output_path gen.wav \
  --model_dir checkpoints \
  --config checkpoints/config.yaml \
  --fp16 \
  --force
```

### Python API
IndexTTS2 can be embedded in Python workflows for fine-grained control over emotional style, randomization, and duration.

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=False,
    use_cuda_kernel=False,
    use_deepspeed=False,
)

tts.infer(
    spk_audio_prompt="examples/voice_07.wav",
    text="Please read this sentence with a calm, thoughtful tone.",
    emo_audio_prompt="examples/emo_sad.wav",
    emo_alpha=0.8,
    output_path="gen.wav",
    verbose=True,
)
```

Additional examples covering emotion vectors, textual style prompts, and random sampling are available in [`docs/README_zh.md`](docs/README_zh.md).

## Repository layout
- `webui.py` – Gradio-based graphical interface for zero-shot inference and audio export.
- `indextts/` – Core inference modules, GPT acoustic model, BigVGAN vocoder, utilities, and the CLI entry point.
- `examples/` – Reference voices, emotion prompts, and scripted cases for quick testing.
- `tools/` – Diagnostic helpers including GPU capability checks.
- `docs/README_zh.md` – Comprehensive usage guide in Simplified Chinese.

## Community & support
- QQ groups: 553460296 (Group 1) / 663272642 (Group 4)
- Discord: <https://discord.gg/uT32E7KDmy>
- Email: <indexspeech@bilibili.com>

> ⚠️ Only the GitHub organization at <https://github.com/index-tts/index-tts> is maintained by the core IndexTTS team. Other distributions or services are unofficial and not endorsed.

## Citation
If you build upon IndexTTS2 in academic work, please cite the accompanying paper:

```bibtex
@article{IndexTTS2,
  title   = {IndexTTS2: Expressive and Duration-Controlled Zero-Shot Text-to-Speech},
  author  = {Index Team},
  journal = {arXiv},
  year    = {2025},
  eprint  = {2506.21619},
  url     = {https://arxiv.org/abs/2506.21619}
}
```
