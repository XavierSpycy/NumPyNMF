# Non-negative Matrix Factorization using NumPy

## 1. :sparkles: Introduction
NMF aims to decipher the following formula:

$$X \approx D R$$

<p align="center">
  <img src="figures/NMFs.png">
  <br>
  Figure 1. Illustration of NMF
</p>

Our experiments seek to compare the robustness of various NMF variants.

- Datasets: ORL, Cropped YaleB              
- NMFs			   
- Noise Types

## 2. :sparkles: NMF Variants
## 3. :sparkles: Noise Types
## 4. :sparkles: Convergence Trends

## 5. :sparkles: Results
### 5.1 Metrics
#### 5.1.3 What are They?
- Root Means Square Errors (RMSE)     
$\mathrm{RMSE} = \lVert \mathbf{X - DR} \rVert^2_F$
- Average Accuracy      
$\mathrm{Acc(Y, Y_{pred})} = \frac{1}{n} \sum_{\substack{i}}^n \{\mathrm{{Y_{(pred)(i)}}} = \mathrm{Y(i)}\}$
- Normalized Mutual Information (NMI)     
$\mathrm{NMI(Y, Y_{pred})} = \frac{2 \times I(\mathrm{Y, Y_{pred}})}{H(\mathrm{Y}) + H(\mathrm{Y_{pred})}}$,      
where $I(\cdot, \cdot$) is the mutual information, $H(\cdot)$ is the entropy.


### 5.2 Reconstruction Effects
<p align="center">
  <img src="figures/yaleb-gau-recon.png">
  <br>
  Figure 11. 
</p>

<p align="center">
  <img src="figures/yaleb-lap-recon.png">
  <br>
  Figure 12. 
</p>

<p align="center">
  <img src="figures/yaleb-uniform-recon.png">
  <br>
  Figure 13. 
</p>

<p align="center">
  <img src="figures/yaleb-block-10-recon.png">
  <br>
  Figure 14. 
</p>

<p align="center">
  <img src="figures/yaleb-block-20-recon.png">
  <br>
  Figure 15. 
</p>

## 6. :sparkles: Project Structure
```
├── code/
│   ├── algorithm/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── decomposition.py
│   │   ├── evaluations.py
│   │   ├── intialization.py
│   │   ├── label.py
│   │   ├── NMF.py
│   │   ├── preprocess.py
│   │   ├── sample.py
│   │   └── visualize.py
│   └── data/
│       ├── CroppedYaleB/
│       └── ORL/
├── figures/ ... # Some visualizations
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── run.py

```