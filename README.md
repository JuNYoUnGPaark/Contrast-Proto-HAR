# Contrastive Prototype–Guided CrossFormer for Sensor-Based Human Activity Recognition

<p align="center"><img src='./overall.png'></p>

This repository implements the methodology proposed in the paper "Contrastive Prototype–Guided CrossFormer for Sensor-Based Human
Activity Recognition".


## Paper Overview
**Abstract**: Despite significant progress in sensor-based Human Activity Recognition (HAR), relying solely on a classification loss often results in a weakly constrained embedding geometry. This limitation makes it challenging to effectively separate look-alike activities and ultimately leads to performance degradation. To address this, we propose a novel Contrastive Prototype-Guided Framework that injects crucial class-level priors. Our method first applies channel- and temporal-attentive filtering to prioritize informative sequence regions. It then leverages a CrossFormer block to effectively capture long-range and multi-scale temporal dependencies. Finally, it aligns sample features with a learnable prototype bank using cross-attention coupled with a contrastive objective. By explicitly promoting sample–prototype consistency, the proposed framework produces more discriminative and compact activity representations. This not only leads to better robustness against typical wearable perturbations but also remains efficient for on-device deployment. The model achieves state-of-the-art results on four widely used public benchmarks, attaining F1 scores of 0.9881 (UCI-HAR), 0.9709 (PAMAP2), 0.9886 (MHEALTH), and 0.9910 (WISDM). Furthermore, the design is highly lightweight and real-time, featuring only 0.082M parameters, 5.7M FLOPs, and a 2 ms average inference time, fully supporting on-device deployment.

## Dataset
- UCI-HAR dataset is available at https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- PAMAP2 dataset is available at https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- MHEALTH dataset is available at https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- WISDM dataset is available at https://www.cis.fordham.edu/wisdm/dataset.php

## Requirements
```
torch==2.6.0+cu126
numpy==2.3.4
pandas==2.3.3
scikit-learn==1.7.2
matplotlib==3.9.2
seaborn==0.13.2
fvcore==0.1.5.post20221221
```
To install all required packages:
```
pip install -r requirements.txt
```

## Codebase Overview
- `model.py` - Implementation of the proposed Contrast-CrossFormer+CBAM architecture with prototype-guided contrastive head.
The implementation uses PyTorch, Numpy, pandas, scikit-learn, matplotlib, seaborn, and fvcore (for FLOPs analysis).

## Citing this Repository

If you use this code in your research, please cite:

```
@article{Contrastive Prototype–Guided CrossFormer for Sensor-Based Human
Activity Recognition,
  title = {Contrastive Prototype–Guided CrossFormer for Sensor-Based Human
Activity Recognition},
  author={JunYoung Park and Myung-Kyu Yi}
  journal={},
  volume={},
  Issue={},
  pages={},
  year={}
  publisher={}
}
```

## Contact

For questions or issues, please contact:
- JunYoung Park : park91802@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
