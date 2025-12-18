````
# TDFusion
Codes for ***CNN-MAMBA NETWORK WITH CLIP-BASED SEMANTIC ALIGNMENT FOR ANATOMICAL AND FUNCTIONAL IMAGE FUSION***

## Abstract

The fusion of anatomical and functional images is critical for accurate medical diagnosis. However, existing deep 
learning methods often struggle to preserve fine anatomical details,capture global dependencies, and incorporate effective semantic constraints. To address these limitations, we propose
CLIP-Mamba, a novel framework integrating three key components: (1) the Differential-Partial Focus Module (DPFM) for extracting critical local structural and functional features;(2) the Detail-Enhanced Mamba Module (DEMM) for modeling long-range dependencies and enhancing global representations;
and (3) the Spatial-Channel Reconstruction Module (SCRM) for suppressing redundancy and reconstructing complementary features. Additionally, we introduce CLIP-based
semantic alignment to enforce consistency between structural and functional semantics. Experiments on MRI-SPECT datasets demonstrate that CLIP-Mamba preserves anatomical
boundaries and highlights functional signals, achieving superior semantic consistency and detail representation. T
## Update
- [2025/09] Release the code.

## Citation

```

## 🌐 Usage

### 🏊 Training
**1. Data Preparation**

Harvard medical website,  http://www.med.harvard.edu/AANLIB/home.html

**2. Commence Training**

```
python train.py
``` 

### 🏄 Testing

**1. Pretrained models**

Pretrained models are available in ``./models/model_300.pth``, which are the models trained on Harvard datasets.

**2. Test cases**

Running 
```
python test.py
``` 
will fuse these cases, and the fusion results will be saved in the 'output' folder.

````