# SMiBPT: Entropy-Aware Byte Patching Transformer for Molecular Representation Learning

A novel entropy-aware Transformer-based architecture for molecular representation learning. SMiBPT introduces **dynamic byte-level patching**, **chemical motif awareness**, and **adaptive masked language modeling** to effectively capture semantically rich substructures in both SMILES and DeepSMILES molecular encodings.

---

## 🌟 Key Features

- **Entropy-Aware Byte Patching**: Dynamically adjusts patch size based on local sequence entropy.
- **Chemical Motif Integration**: Gives special attention to aromatic rings, charged groups, and metals.
- **Rotary Positional Encoding (RoPE)**: Enhances sequence modeling of untruncated molecules.
- **Adaptive Masked Language Modeling (MLM)**: Prioritizes masking in high-entropy, chemically informative regions.
- **Pretrained on ~216M molecules** from PubChem.
- **Supports SMILES & DeepSMILES** formats.
- **Fine-tuned on MoleculeNet** benchmarks: BBBP, ESOL, Lipophilicity.
- **Fine-tuned further on new real-world datasets** benchmarks: Covid-19 and Antimalarial.


---

## 🧬 Applications

- Molecular property prediction (classification & regression)
- Structure–activity relationship modeling
- Drug discovery and cheminformatics research
- Pretraining and zero-shot transfer between SMILES ↔ DeepSMILES

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/SMiBPT.git
cd SMiBPT
pip install -r requirements.txt
````

**Requirements include**:

* Python ≥ 3.8
* PyTorch ≥ 1.11
* RDKit
* scikit-learn
* tqdm
* transformers (optional for comparison)

---

## 📁 Project Structure

```
SMiBPT/
│
├── model/                   # Transformer and embedding architecture
│   ├── smibpt_model.py      # SMILESMLMTransformer definition
│   ├── patch_embedding.py   # DynamicBytePatchEmbedding
│   ├── rope.py              # Rotary Positional Embedding (RoPE)
│
├── utils/                  
│   ├── entropy.py           # Entropy computation functions
│   ├── motifs.py            # Aromatic, charged, metal motif detection
│   ├── masking.py           # Adaptive MLM masking functions
│   ├── evaluation.py        # AUC, F1, RMSE, R² calculations
│
├── data/                    # SMILES datasets and MoleculeNet benchmarks
│   ├── smiles_dataset.py    # Dataset class with adaptive byte patching
│   ├── collate_fn.py
│
├── train_pretrain.py        # Main script for MLM pretraining
├── train_finetune.py        # Fine-tuning for classification/regression
├── cross_validate.py        # 10-fold CV pipeline for downstream tasks
├── config.yaml              # Editable training settings
├── README.md
└── requirements.txt
```

---

## 🔁 Pretraining

```bash
python train_pretrain.py --config config.yaml
```

This pretrains SMiBPT on untruncated SMILES/DeepSMILES using entropy-aware masking and byte patching. By default, it logs loss, accuracy, and entropy dynamics per epoch.

If you would like to request access to the pre-training dataset, please contact us: `medardedmund25@chungbuk.ac.kr`

---

## 🧪 Fine-Tuning on Downstream Tasks

To fine-tune on BBBP (classification):

```bash
python train_finetune.py --task BBBP --mode classification --folds 10
```

To fine-tune on ESOL (regression):

```bash
python train_finetune.py --task ESOL --mode regression --folds 10
```

Supports MoleculeNet tasks: `BBBP`, `ESOL`, `Lipophilicity`, `Tox21`, etc.

---

## 🧠 Adaptive Masking Example

```python
from masking import mask_byte_patches_adaptive
masked_patches, labels = mask_byte_patches_adaptive(byte_patches, entropy_vals)
```

High-entropy regions (e.g., heterocycles, aromatic rings) are masked more often to improve contextual learning.

---

## 📊 Evaluation Metrics

* **Classification**: AUC, F1 (10-fold CV)
* **Regression**: RMSE, R²
* Epoch-wise logging and best-model saving (`best_model.pth`) are supported.

---

## 📈 Pretrained Model

Download pretrained SMiBPT weights:

* `best_model.pth`: Pretrained on \~216M molecules with entropy-aware patching and RoPE

```python
model.load_state_dict(torch.load("best_model.pth"))
```

---

## 📚 Citation

If you use SMiBPT in your research, please cite:

```bibtex
@article{mswahili2025smibpt,
  title={Entropy-based byte patching transformer for self-supervised pretraining of SMILES data},
  author={Mswahili, Medard Edmund and Hwang, JunHa and Jo, Kyuri and Rajapakse, Jagath C and Kov{\'a}cs, P{\'e}ter and Jeong, Young-Seob},
  journal={TBD (submitted)},
  year={2025}
}
```

---

## ✨ Acknowledgements

This work builds upon molecular language modeling techniques and explores adaptive patching and entropy-driven structural learning. We thank the contributors to RDKit, MoleculeNet, and PubChem for enabling data-driven research.

---

## 💬 Questions or Feedback?

Open an issue or contact us at `medardedmund25@chungbuk.ac.kr`.
