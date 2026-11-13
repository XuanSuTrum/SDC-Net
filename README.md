# 🧠 SDC-Net  
## A Domain Adaptation Framework with Semantic-Dynamic Consistency for Cross-Subject EEG Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)]
[![License](https://img.shields.io/badge/License-Academic-green.svg)]

Official implementation of **SDC-Net**, a novel unsupervised domain adaptation framework designed for robust cross-subject EEG emotion recognition in affective brain-computer interfaces (aBCIs).

---

## 🔥 Highlights

✅ Fully **label-free domain adaptation** for EEG emotion recognition  
✅ Semantic-aware **dynamic distribution alignment in RKHS**  
✅ Confidence-guided pseudo-label learning  
✅ Dual-domain similarity consistency modeling  
✅ Strong robustness against **negative transfer**  

---

## 🧾 Paper

> **SDC-Net: A Domain Adaptation Framework with Semantic-Dynamic Consistency for Cross-Subject EEG Emotion Recognition**

📌 **Authors:** Jiahao Tang, Youjun Li, Xiangting Fan, Yangxuan Zheng, Siyuan Lu, Xueping Li, Peng Fang, Chenxi Li*, Zi-Gang Huang*

📌 **Affiliation:** Xi’an Jiaotong University  

📌 **Paper:** *(https://arxiv.org/abs/2507.17524)*  

---

## 🚀 Quick Start

After preparing the dataset and installing dependencies, you can launch the training process with a single command:

```bash
python main_5.py
```
If you encounter any issues when running the code, please feel free to open an issue.


## ANML pilot branch

This branch adds a controlled A/B/C exploratory experiment. Start with `run_anml_pycharm.py` (data audit first), then follow [the Chinese PyCharm guide](docs/ANML_PILOT.md). A disables relation learning, B uses a repaired DSCL baseline, and C uses an independent DANML equation (22) adaptation. No training results are claimed; the experimental baseline includes documented common fixes and is not a numerical reproduction of the original paper.
