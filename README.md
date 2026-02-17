# AnchorWeave: World-Consistent Video Generation with Retrieved Local Spatial Memories

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://zunwang1.github.io/AnchorWeave)  [![Paper](https://img.shields.io/badge/Paper-PDF-red)](#)  [![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/wz0919/AnchorWeave)

**Zun Wang**<sup>1</sup>, **Han Lin**<sup>1</sup>, **Jaehong Yoon**<sup>2</sup>, **Jaemin Cho**<sup>3</sup>, **Yue Zhang**<sup>1</sup>, **Mohit Bansal**<sup>1</sup>

<sup>1</sup>University of North Carolina, Chapel Hill · <sup>2</sup>NTU Singapore · <sup>3</sup>AI2

## Abstract

Maintaining spatial world consistency over long horizons remains a central challenge for camera-controllable video generation. Existing memory-based approaches often condition generation on globally reconstructed 3D scenes by rendering anchor videos from the reconstructed geometry in the history. However, reconstructing a global 3D scene from multiple views inevitably introduces cross-view misalignment, as pose and depth estimation errors cause the same surfaces to be reconstructed at slightly different 3D locations across views. When fused, these inconsistencies accumulate into noisy geometry that contaminates the conditioning signals and degrades generation quality.

We introduce **AnchorWeave**, a memory-augmented video generation framework that replaces a single misaligned global memory with multiple clean local geometric memories and learns to reconcile their cross-view inconsistencies. To this end, AnchorWeave performs coverage-driven local memory retrieval aligned with the target trajectory and integrates the selected local memories through a multi-anchor weaving controller during generation. Extensive experiments demonstrate that AnchorWeave significantly improves long-term scene consistency while maintaining strong visual quality, with ablation and analysis studies further validating the effectiveness of local geometric conditioning, multi-anchor control, and coverage-driven retrieval.

---

## TODO

- [x] CogVideoX training code
- [ ] Processed training data
- [ ] Data processing scripts
- [ ] Inference example
- [ ] Wan-based code

---

## Setup

### 1. Clone & Environment

```bash
git clone https://github.com/wz0919/AnchorWeave.git
cd AnchorWeave
conda create -n anchorweave python=3.10
conda activate anchorweave
pip install -r requirements.txt
```

### 2. Download Models

Place **CogVideoX-5B-I2V** under `./pretrained/CogVideoX-5b-I2V/`:

```bash
# Download from https://github.com/THUDM/CogVideo
mkdir -p pretrained
# Place CogVideoX-5b-I2V folder in pretrained/
```

---

## Training

Edit `scripts/train_with_latent.sh` to set `video_root_dir` and `output_dir`, then:

```bash
# Edit GPU config in training/accelerate_config_machine.yaml (num_processes)
bash scripts/train_with_latent.sh
```

---

## Inference

```bash
# Place example dataset under ./data/example_dataset, then:
bash scripts/inference.sh
```

---

## Acknowledgements

- [CogVideoX](https://github.com/THUDM/CogVideo)
- [CogVideoX-ControlNet](https://github.com/TheDenk/cogvideox-controlnet)

---

## Citation

```bibtex
@article{anchorweave2025,
  title={AnchorWeave: World-Consistent Video Generation with Retrieved Local Spatial Memories},
  author={Wang, Zun and Lin, Han and Yoon, Jaehong and Cho, Jaemin and Zhang, Yue and Bansal, Mohit},
  year={2025}
}
```
