English| [中文版](README.zh-CN.md)

<div align="center" style="font-weight: bold;">
  <a href="app.py">Demo</a>
</div>
<div align="center" style="font-weight: bold;">
  Running...
</div>
<p align="center">
  <img src="figures/app_screen_shoot.jpg">
</p>
<div align="center" style="font-weight: bold;">
  Results
</div>
<p align="center">
  <img src="figures/app_screen_shoot_2.jpg">
</p>

Dive into the heart of our technology with a hands-on demonstration. Simply initiate the process in your terminal:

```bash
$ python app.py
```

Engage directly with our algorithm in a few easy steps: 
- Step 1. Select your desired parameters on the left side; 
- Step 2. Click `Execute Algorithm`; 
- Step 3. Eagerly await your customized results on the right. Experience our solutions at your fingertips.

# Non-negative Matrix Factorization using NumPy
- [Quick Start](#rocket-quick-start)
- [1. Introduction](#1-sparkles-introduction)
- [2. NMF Variants](#2-sparkles-nmf-variants)
- [3. Noise Types](#3-sparkles-noise-types)
- [4. Setup and Execution](#4-sparkles-setup-and-execution)
- [5. Convergence Trends](#5-sparkles-convergence-trends)
- [6. Results](#6-sparkles-results)
  - [6.1. Metrics](#61-metrics)
    - [6.1.1. What are They?](#611-what-are-they)
    - [6.1.2. Why RMSE More Important?](#612-why-rmse-more-important)
  - [6.2. Performance on ORL and YaleB Datasets](#62-performance-on-orl-and-yaleb-datasets)
  - [6.3. Reconstruction Effects](#63-reconstruction-effects)
- [7. Project Structure](#7-sparkles-project-structure)
- [8. Updata Log & TODO List](#8-update-log--todo-list)
- [9. Contribution](#9-handshake-contribution)

:pushpin: **Important Notice**:
Please ensure that dataset files are placed in the `data` directory before executing `run.py`. For emphasis, we've incorporated an error notification mechanism. Moreover, we've provided comprehensive docstrings and comments within our code. Should you have any questions, feel free to explore our source code in depth.

Please refrain from intentionally inputting unexpected data types to test our algorithms. We do not have initial input type assertions, so they cannot reject inappropriate inputs from the start. Thank you for your understanding.

## :rocket: Quick Start
1. Simplicity

To swiftly experience the method in action, simply configure and run the following in `run.py`:
```python
from algorithm.pipeline import Pipeline

pipeline = Pipeline(nmf='L1NormRegularizedNMF', # Options: 'L2NormNMF', 'L1NormNMF', 'KLDivergenceNMF', 'ISDivergenceNMF', 'L21NormNMF', 'HSCostNMF', 'L1NormRegularizedNMF', 'CappedNormNMF', 'CauchyNMF'
                    dataset='YaleB', # Options: 'ORL', 'YaleB'
                    reduce=3, # ORL: 1, YaleB: 3
                    noise_type='salt_and_pepper', # Options: 'uniform', 'gaussian', 'laplacian', 'salt_and_pepper', 'block'
                    noise_level=0.08, # Uniform, Gassian, Laplacian: [.1, .3], Salt and Pepper: [.02, .10], Block: [10, 20]
                    random_state=99, # 0, 42, 99, 512, 3407 in our experiments
                    scaler='MinMax') # Options: 'MinMax', 'Standard'

# Run the pipeline
pipeline.run(max_iter=500, verbose=True) # Parameters: max_iter: int, convergence_trend: bool, matrix_size: bool, verbose: bool
pipeline.evaluate(idx=9, imshow=True) # Parameters: idx: int, imshow: bool
```

Our development framework empowers you to effortlessly create your own NMF algorithms with minimal Python scripting. Here's how you can get started:
```python
import numpy as np
from algorithm.nmf import BasicNMF
from algorithm.pipeline import Pipeline

class ExampleNMF(BasicNMF):
    # To tailor a unique NMF algorithm, subclass BasicNMF and redefine matrix_init and update methods.
    def matrix_init(self, X, n_components, random_state=None):
        # Implement your initialization logic here.
        # Although we provide built-in methods, crafting a bespoke initialization can markedly boost performance.
        # D, R = <your_initialization_logic>
        # D, R = np.array(D), np.array(R)
        return D, R  # Ensure D, R are returned.

    def update(self, X, **kwargs):
        # Implement the logic for iterative updates here.
        # Modify self.D, self.R as per your algorithm's logic.
        # flag = <convergence_criterion>
        return flag  # Return True if converged, else False.
```

Test the initial performance of your algorithm seamlessly with our pipeline:
```python
pipeline = Pipeline(nmf=ExampleNMF(),
                    dataset='YaleB',
                    reduce=3,
                    noise_type='salt_and_pepper',
                    noise_level=0.08,
                    random_state=99,
                    scaler='MinMax')
```
For a comprehensive assessment, conduct experiments across our datasets:
```python
from algorithm.pipeline import Experiment

exp = Experiment()
# Once you build the data container
# You can choose an NMF algorithm and execute the experiment
exp.choose('L1NormRegularizedNMF')
# This step is very time-consuming, please be patient.
# If you achieve a better performance, congratulations! 
# You can share your results with us.
# Similarly, you can replace 'L1NormRegularizedNMF' with other your customized NMF algorithm
exp.execute()
```
Note: The `Experiment` function accepts either a string representing a built-in algorithm or a `BasicNMF` object, enabling you to directly evaluate your custom NMF algorithm.

2. Convenience

You are invited to try out our experiments on Google Colab. First, execute all the code snippets in the `Setup` section to access our repository. ~~Also, all you need to do is upload the `data.zip` file~~. (You no longer need to upload the `data.zip` file.)

Once the experimental environment is set up, you have the choice to either run `run.py` in the terminal or adjust the default settings and then execute the script within the Jupyter notebook as you see fit.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/XavierSpycy/NumPyNMF/blob/main/run.ipynb)

For a more comprehensive setup and detailed execution instructions, please refer to the "Setup and Execution" section.

## 1. :sparkles: Introduction
It is a matrix factorization technique where all the elements of the decomposed matrices are required to be non-negative. This kind of decomposition is especially suitable for datasets where all the elements are non-negative, such as image data or text data.

NMF aims to decipher the following formula:

$$X \approx D R$$

<p align="center">
  <img src="figures/NMFs.png">
  <br>
  Figure 1. Illustration of NMF
</p>

Where, if 
$X$ is of size $m \times n$, typically $D$ would be of size $m \times k$ and $R$ would be of size $k \times n$, where $k$ is a predefined number of factors and is usually less than both $m$ and $n$.

NMF has found utility in many applications, including feature extraction, image processing, and text mining.

Our experiments seek to compare the robustness of various NMF variants.

<p align="center">
  <img src="figures/pin.png">
  <br>
  Figure 2. Illustration of Our Experiments
</p>

- **2** Datasets: ORL, Cropped YaleB              
- **8** NMFs: [$L_2$ Norm Based](https://www.nature.com/articles/44565), [KL Divergence](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf), [IS Divergence](https://watermark.silverchair.com/neco.2008.04-08-771.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA8swggPHBgkqhkiG9w0BBwagggO4MIIDtAIBADCCA60GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMWw-7AxuDs18JkL0UAgEQgIIDfuk3ZMzOzVoJFC51RfL5EJVANHo0W7nrAuUIXGE4nA-F_Y-Ftsv7BDxQoIBREl79JoQUKTkNaUTT_1Hn7yjsQVxDx_SncE8Z9G0TZrwmMyZ-n45oJ2hIx00vSYab4CWUkpdmtR3CcHVVjxw7vyDMAtBN04C9_i58D8rZjUnPnhcDq2bQtsCjb6m8bzCOuaibcJlp3FcViwOmRMUjvtwQBJxL4cDsE4dvRkA5-T0uERnpdLzJPHDWX5riECVVq2N7THuG6mJzEiIEpRPep6ac-YUpqGCCeWTcefsDx3Yy8EJ49CVClxJP377N3YV9S8oeTWZhFATF17HkIfN7cKkCNRwCCLrm_gqIDJYMe4DMItKLGtcWJ4KDbQ-MtT-ufzvw9EuQLPsYXPMEu2wDsEXAeFD7F4SCfmVydbfnc72fyd1yMfSUHRZmMHN-Wj8g9I64zEFV8SHlIXuoMk7Mt5SGJdPooPdgNQCZ00mzIOxmcDxk7Rm0mNAjoaZ26RqP1kD34MZq9mlh1kJ2S6iBSB695ne3s20r0Yn5FhlFtAAOFYetRpzcWRYUZROlIVanBgz8HYG_ROViNSuPuY6tZN579t3tgoHAwDa1b8dqTPLcq-jWGM5du79adPaFcYtGtFEup9KZuAt4eIJfiDPkuEym_Uq_YWaUoYOCExxA3VPfJn2dJp6yJNjgHKfjQg98BeLt341w7DAiY_d31ZCfRk-l7surQhc2RY62EJp_7VsEB45KdqyY9ukjG8pnpj8niDhGoLTNiA_78Tb-dZ_tFV4wd99WjqmMC1CwMcInWZZwWOt22zaX4lM02itiuLtHS9Q3ZudAqrSWtOTPT2OSylAL1AzXOharFAOQNsraqvFqFeF2VuQFPr0Pz9ethHvBDqqM89gIURSim5sVe4-PitmYTvLgk-9FBiWOLvreFkkAY9QXeoaYR05q_bBshxYgOFT6-ndKDFo2zwDQ1zGtZPKlhkBUx6PDwHTiIvKFIxOGu9GCJHsBLFL7VUfr1Me_JZrd9laEmcbkstyRrGp-QHa80x7zO0e761WtINWjZgW0peoU4C1rl1yD8j-y6X5jAGh6xgpsLg5OHm0WRJbu2KM4VYkYE7Q083aybLYxo7O-f3WoPw5byvHvxmrmLYtTJLUZQOkFM_VzoaukZjkP3O63czUuiVbNodMgfy4S1ym92Q), [$L_{2, 1}$ Norm Based](https://dl.acm.org/doi/pdf/10.1145/2063576.2063676?casa_token=iEFm3h7S8soAAAAA:1TTJQgWGYV3KLmLDrSg_F3QRrfqR0NTjUp7mT4PzHFMXAd1jvI3X1YMk0ScPjuXAWDGS4gBLLUdxeg), [Hypersurface Cost](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6483162&casa_token=yoHaNL_ZqHYAAAAA:foV5TTg1uyyu6YclDnjrH38G4o3BsJdYHIN0uHwZOtuAead2M2pYGok7Odu-uv2IVJABIBC1DA), [$L_1$ Norm Regularized](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=dfd42efb5db96cba415ceb69c21b75f7930bbdde), [Capped Norm Based](https://dl.acm.org/doi/pdf/10.1145/2806416.2806568?casa_token=ixpoEsWLvA0AAAAA:7sl6K4UL2Klnib2WsOR7Sb8-mhKFqszmQKrCIlRp8TLLj941O-2eC16W79Miw8a1yPPu8_N7fj7a7g), [Cauchy](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7336900&casa_token=T3zhvmGNJpcAAAAA:Gu0xKMZiWnC0ecq0-8UPFpwpxvyow_OvySK7ceJTy7e3FWY9eXHORAgVITAZUtTx1TasuOJRPA)              
- **5** Noise Types: Uniform, Gaussian, Laplacian, Block Occlusion, Salt and Pepper

## 2. :sparkles: NMF Variants
**Note**: GitHub itself does not support rendering LaTeX math formulas in Markdown previews, so some formulas may not display correctly. You might want to use other tools to view these formulas.

- $L_2$ Norm Based NMF
  - Cost Function:
  $\lVert X - DR \rVert^2 = \sum_{\substack{ijk}}(x_{ij} - d_{ik}r_{kj})^2$
  - Update Rule:          
  $\mathbf{D} \leftarrow \mathbf{D} \times \frac{\mathbf{X} \mathbf{R^\top}}{\mathbf{D} \mathbf{R} \mathbf{R^\top}}\\   
  \mathbf{R} \leftarrow \mathbf{R} \times \frac{\mathbf{D^\top} \mathbf{X}}{\mathbf{D^\top} \mathbf{D} \mathbf{R}}$

- KL Divergence NMF
  - Cost Function:      
  $d_{KL}(\mathbf{X} \lVert \mathbf{DR}) = \sum_{\substack{ijk}}(x_{ij}\mathrm{log}\frac{x_{ij}}{d_{ik}r_{kj}} - x_{ij} + d_{ik}r_{kj})$
  - Update Rule:      
  $\mathbf{D} \leftarrow \mathbf{D} \times \frac{(\frac{\mathbf{X}}{\mathbf{DR}})\mathbf{R}^\top}{\mathbf{1}_{m,n} \cdot \mathbf{R}^\top}\\
  \mathbf{R} \leftarrow \mathbf{R} \times \frac{\mathbf{D}^\top \left(\frac{\mathbf{X}}{\mathbf{DR}} \right)}{\mathbf{D}^\top \cdot \mathbf{1}_{m,n}}$

- IS Divergence NMF
  - Cost Function:      
  $d_{IS}(\mathbf{X} \lVert \mathbf{DR}) = \frac{\mathbf{X}}{\mathbf{DR}} - \mathrm{log}\frac{\mathbf{X}}{\mathbf{DR}} - 1$
  - Update Rule:      
  $\mathbf{D} \leftarrow \mathbf{D} \times \frac{((\mathbf{DR}^{-2}) \mathbf{X})\mathbf{R}^\top}{(\mathbf{DR})^{-1} \mathbf{R}^\top}\\
  \mathbf{R} \leftarrow \mathbf{R} \times \frac{\mathbf{D}^\top ((\mathbf{DR})^{-2}\mathbf{X})}{\mathbf{D}^\top (\mathbf{DR})^{-1}}$

- $L_{2, 1}$ Norm Based NMF
  - Cost Function:      
  $\lVert \mathbf{X - DR} \rVert_{2, 1} = \sum_{\substack{i=1}}^n \sqrt{\sum_{\substack{j=1}^p}(\mathbf{X} - \mathbf{DR})_{ji}^2}  = \sum_{\substack{i=1}}^n \lVert x_i - \mathbf{D}r_i \rVert$
  - Update Rule:      
  $D_{ji} \leftarrow D_{jk} \times \frac{(\mathbf{X \Lambda R^\top})_{jk}}{(\mathbf{DR\Lambda R^\top})_jk} \\
  R_{ki} \leftarrow R_{ki} \times \frac{(\mathbf{D^\top X\Lambda})_{ki}}{(\mathbf{D^\top DR\Lambda})_{jk}}\\
  $
  where $\Lambda$ is a diagonal matrix with the diagonal elements given by,     
  $D_{ii} = \frac{1}{\sqrt{\sum_{\substack{j=1}}^p(\mathbf{X - DR})_{ji}^2}} = \frac{1}{\lVert x_i - \mathbf{D}r_i \rVert}$

- Hypersurface Cost NMF
  - Cost Function:
  $\phi(\mathbf{D}, \mathbf{R}) = \frac{1}{2}(\sqrt{1 + \lVert \mathbf{X} - \mathbf{DR} \rVert^2} - 1)$
  - Update Rule:      
  $\mathbf{D} \leftarrow \mathbf{D} - \alpha\frac{\mathbf{DRR}^{\top} - \mathbf{XR}^{\top}}{\sqrt{1 + \lVert \mathbf{X} - \mathbf{DR} \rVert}}\\
  \mathbf{R} \leftarrow \mathbf{R} - \beta \frac{\mathbf{D}^{\top}\mathbf{DR} - \mathbf{D}^{\top}\mathbf{X}}{\sqrt{1 + \lVert \mathbf{X} - \mathbf{DR} \rVert}}$

- $L_1$ Norm Regularized NMF
  - Cost Function:
  $\lVert \mathbf{X} - \mathbf{DR} - \mathbf{S}\rVert_F^2 + \lambda \lVert S \rVert_1$
  - Update Rule:      
  $\mathbf{S} \leftarrow \mathbf{X} - \mathbf{DR}\\
  \mathbf{S}_{ij} \leftarrow 
    \begin{cases}
        \mathbf{S}_{ij} - \frac{\lambda}{2} \text{  , if} \mathbf{S}_{ij} > \frac{\lambda}{2} \\
        \mathbf{S}_{ij} + \frac{\lambda}{2} \text{  , if} \mathbf{S}_{ij} < \frac{\lambda}{2}\\
        0 \text{    , otherwise}
    \end{cases}\\
    \mathbf{D} \leftarrow \frac{\left | (\mathbf{S} - \mathbf{X})\mathbf{R}^{\top}\right | - ((\mathbf{S} - \mathbf{X})\mathbf{R}^{\top}}{2\mathbf{DRR}^{\top}}\\
    \mathbf{R} \leftarrow \frac{\left |\mathbf{D}^{\top}(\mathbf{S} - \mathbf{X})\right | - (\mathbf{D}^{\top}(\mathbf{S} - \mathbf{X})}{2\mathbf{D^{\top}}\mathbf{DR}}\\
    \mathbf{D} \leftarrow \frac{\mathbf{D}}{\sqrt{\sum^n_{k=1}\mathbf{D}_{kj}^2}}\\
    \mathbf{R} \leftarrow \mathbf{R}\sqrt{\sum^n_{k=1}\mathbf{D}_{kj}^2}$

- Capped Norm Based NMF
  - Update Rule:     
  $\mathbf{D} \leftarrow \mathbf{D}\frac{\mathbf{XIR}^\top}{\mathbf{DRIR}^{\top}}\\
  \mathbf{R} \leftarrow \mathbf{R}\sqrt{\frac{\mathbf{IXD}}{\mathbf{IR}^{\top}\mathbf{RXD}}}\\
  \mathbf{I}_{jj} = 
    \begin{cases} \frac{1}{2\lVert x_j - \mathbf{D}r_j\rVert_2}\text{   , if} \lVert x_j - \mathbf{D}r_j\rVert \leq \theta \\
    0 \text{    , otherwise}
    \end{cases}$,      
    where $\mathbf{I}$ is initialized as an identify mamtrix and then will be updated to a diagonal matrix.

- Cauchy NMF
  - Update Rule:         
  $\theta \leftarrow \theta \cdot \frac{b_\theta}{a_\theta + \sqrt{a_\theta^2 + 2b_\theta \cdot a_\theta}}$     
  For $\mathbf{D}$,     
  $a_\theta =  \frac{3}{4} \frac{\sigma}{\sigma^2 + \mathbf{X}} \mathbf{R}^\top\\
  b_\theta = \sigma^{-1}\mathbf{R}^\top$;     
  For $\mathbf{R}$,     
  $a_\theta = \frac{3}{4}\mathbf{D}^{\top}\frac{\sigma}{\sigma^2 + \mathbf{X}}\\
   b_\theta = \mathbf{D}^{\top}\sigma^{-1}$

## 3. :sparkles: Noise Types
- Uniform:
<p align="center">
  <img src="./figures/uniform_noise.png">
  <br>
  Figure 3. Uniform Noise
</p>

- Gaussian
<p align="center">
  <img src="./figures/gaussian_noise.png">
  <br>
  Figure 4. Gaussian Noise
</p>

- Laplacian
<p align="center">
  <img src="./figures/laplacian_noise.png">
  <br>
  Figure 5. Laplacian Noise
</p>

- Block Occlusion
<p align="center">
  <img src="figures/block_noise.png">
  <br>
  Figure 6. Block Noise
</p>

- Salt and Pepper
<p align="center">
  <img src="figures/salt_and_pepper_noise.png">
  <br>
  Figure 7. Salt and Pepper Noise
</p>

## 4. :sparkles: Setup and Execution
### Step 1. Environmental Setup
**If you're not concerned about package version conflicts, you may skip this step.**

To avoid potential conflicts between package versions, we ensure smooth execution only under our specified package versions. We can't guarantee flawless operation across all versions of the related packages. However, if you have concerns or wish to ensure the highest compatibility, you can follow the steps below to create a new environment specifically for this experiment.

1. **Create a New Conda Environment:**

   First, you'll want to create a new Conda environment named `NumPyNMF`. To do this, open your terminal or command prompt and enter the following command:

   ```bash
   $ conda create --name NumPyNMF python=3.8
   ```
2. **Activate the New Environment:**

    Before installing any packages or running any scripts, you must activate the `NumPyNMF` environment. To do this, enter the following command:
    ```bash
    $ conda activate NumPyNMF
    ```

3. **Install Required Packages:**

    Navigate to the directory where the `requirements.txt` file is located and install the necessary packages using `pip`:
    ```bash
    $ pip install -r requirements.txt
    ```
4. **Running the Experiment:**

    After setting up the environment and installing the required packages, always ensure that the `NumPyNMF` environment is activated before executing any scripts.

**Important**: As mentioned, we've tailored this environment to avoid potential version conflicts, ensuring optimal compatibility with our codebase. Please use this environment to ensure accurate and conflict-free execution.

### Step 2. Experiment Execution

To run the current experiment, follow these steps:

1. **Configure the Algorithm and Dataset:** In `run.py`, we provide a `Pipeline` class. You can configure your experiment by adjusting its parameters. Here's an explanation of the `Pipeline` parameters:

    - `nmf`: Choose the desired Non-negative Matrix Factorization (NMF) algorithm. Options are: `L2NormNMF`, `L1NormNMF`, `KLdivergenceNMF`, `ISdivergenceNMF`, `RobustNMF`, `HypersurfaceNMF`, `L1NormRegularizedNMF`, `CappedNormNMF`, `CauchyNMF`.
    
    - `dataset`: Select the dataset. Options are: `ORL`, `YaleB`.
    
    - `reduce`: In our experiments, use `1` for `ORL` and `3` for `YaleB`. If the value is too small, the execution time will be excessive; if too large, it will result in information loss.
    
    - `noise_type`: The type of noise. Choices are: `uniform`, `gaussian`, `laplacian`, `salt_and_pepper`, `block`.
    
    - `noise_level`: The level of noise. The specific values vary depending on your choice of noise type.
    
    - `random_state`: The random seed value used in the experiment. In our experiments, we've used: `0`, `42`, `99`, `512`, `3407`.
    
    - `scaler`: The method for data normalization. Choices are: `MinMax`, `Standard`.

2. **Run the Pipeline:**
    ```python
    pipeline.run() 
    ```
    Optional parameters include: `max_iter` (maximum iterations), `convergence_trend` (display convergence trend), `matrix_size` (show matrix size or note), and `verbose` (show training procedure or not).

3. **Evaluate the Results:**
    ```python
    pipeline.evaluate()
    ```
    Optional parameters are: `idx` (index), and `imshow` (display image or not).

### Step 3: Running the Script in Terminal

After you've configured your experiment parameters in the `run.py` script, you can execute the experiment directly from the terminal. Follow these steps:

1. **Navigate to the Directory:**
   
   First, make sure you're in the directory where the `run.py` file is located. Use the `cd` command followed by the directory path to navigate. For example:

   ```bash
   $ cd path/to/your/directory/NumPyNMF
   ```

2. **Execute the Script:**

    Run the script using Python. Depending on your setup, you might use python, python3, or another variation. Here's the general command:
    ```bash
    $ python run.py
    ```

    If you're using Python 3 specifically and have both Python 2 and Python 3 installed, you might use:
    ```bash
    $ python3 run.py
    ```

Hope this helps you carry out the experiment smoothly!

## 5. :sparkles: Convergence Trends
</p>

- $L_1$ Norm Based NMF:
<p align="center">
  <img src="figures/L1conv.png">
  <br>
  Figure 8. Convergence Trend of L<sub>1</sub> Norm Based NMF
</p>


- $L_{2, 1}$ Norm Based NMF:
<p align="center">
  <img src="figures/L21conv.png">
  <br>
  Figure 9. Convergence Trend of L<sub>2, 1</sub> Norm Based NMF
</p>

## 6. :sparkles: Results
### 6.1. Metrics
#### 6.1.1. What are They?
- Root Means Square Errors (RMSE)     
$\mathrm{RMSE} = \sqrt{\frac{1}{N} \lVert \mathbf{X - DR} \rVert^2_F}$
- Average Accuracy      
$\mathrm{Acc(Y, Y_{pred})} = \frac{1}{n} \sum_{\substack{i}}^n \{\mathrm{{Y_{(pred)(i)}}} = \mathrm{Y(i)}\}$
- Normalized Mutual Information (NMI)     
$\mathrm{NMI(Y, Y_{pred})} = \frac{2 \times I(\mathrm{Y, Y_{pred}})}{H(\mathrm{Y}) + H(\mathrm{Y_{pred})}}$,      
where $I(\cdot, \cdot$) is the mutual information, $H(\cdot)$ is the entropy.

#### 6.1.2. Why RMSE More Important?

<p align="center">
  <img src="figures/kl-yaleb-salt.png">
  <br>
  Figure 10. Greater RMSE, Average Accuracy and NMI
</p>

<p align="center">
  <img src="figures/l1normreg-yaleb-salt.png">
  <br>
  Figure 11. Less RMSE, Average Accuracy and NMI
</p>

As illustrated in Figure 10, the reconstructed image exhibits a higher level of granularity.

### 6.2. Performance on ORL and YaleB Datasets
In our preliminary experiments, we observed that certain NMFs might not perform optimally on our specific datasets. This could be attributed to:

- The inherent characteristics of the datasets.
- Potential implementation errors (we leave this to future
work).

We warmly welcome you to delve into our source code and contribute to its enhancement.

<style>
    table, th, td {
        border: 1px solid black;
        text-align: center;
    }
</style>
<table border="1">
    <thead>
        <tr>
            <th rowspan="2">Dataset</th>
            <th rowspan="2">Noise Type</th>
            <th rowspan="2">Noise Level</th>
            <th rowspan="2">Metrics</th>
            <th colspan="9">NMF Algorithm</th>
        </tr>
        <tr>
            <th><i>L<sub>2</sub></i> Norm</th>
            <th>KL Divergence</th>
            <th><i>L<sub>2,1</sub></i> Norm</th>
            <th><i>L<sub>1</sub></i> Norm Regularized</th>
            <th>CappedNorm</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="30">ORL</td>
            <td rowspan="6">Uniform</td>
            <td rowspan="3">0.1</td>
            <td>RMSE</td>
            <td>.1112(.0005)</td>
            <td>.1108(.0005)</td>
            <td>.1116(.0004)</td>
            <td>.1125(.0006)</td>
            <td>.2617(.0035)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.6111(.0394)</td>
            <td>.5911(.0424)</td>
            <td>.5956(.0458)</td>
            <td>.6806(.0275)</td>
            <td>.6883(.0262)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.7696(.0244)</td>
            <td>.7580(.0320)</td>
            <td>.7633(.0295)</td>
            <td>.8210(.0155)</td>
            <td>.8316(.0120)</td>
        </tr>
        <tr>
            <td rowspan="3">0.3</td>
            <td>RMSE</td>
            <td>.2410(.0017)</td>
            <td>.2403(.0018)</td>
            <td>.2411(.0018)</td>
            <td>.2447(.0019)</td>
            <td>.1569(.0011)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.5661(.0126)</td>
            <td>.5650(.0345)</td>
            <td>.5461(.0201)</td>
            <td>.6478(.0168)</td>
            <td>.6639(.0182)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.7450(.0061)</td>
            <td>.7353(.0316)</td>
            <td>.7540(.0262)</td>
            <td>.8051(.0143)</td>
            <td>.8170(.0095)</td>
        </tr>
        <td rowspan="6">Gaussian</td>
            <td rowspan="3">0.05</td>
            <td>RMSE</td>
            <td>.1119(.0140)</td>
            <td>.1116(.0139)</td>
            <td>.1121(.0140)</td>
            <td>.1139(.0139)</td>
            <td>.2699(.0182)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.5683(.0116)</td>
            <td>.5494(.0332)</td>
            <td>.5983(.0472)</td>
            <td>.6750(.0393)</td>
            <td>.6889(.0283)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.7372(.0073)</td>
            <td>.7249(.0233)</td>
            <td>.7540(.0262)</td>
            <td>.8153(.0264)</td>
            <td>.8306(.0131)</td>
        </tr>
        <tr>
            <td rowspan="3">0.08</td>
            <td>RMSE</td>
            <td>.2255(.0380)</td>
            <td>.2249(.0380)</td>
            <td>.2256(.0380)</td>
            <td>.2278(.0377)</td>
            <td>.1710(.0255)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.5706(.0377)</td>
            <td>.5767(.0364)</td>
            <td>.5750(.0434)</td>
            <td>.6389(.0316)</td>
            <td>.6717(.0366)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.7519(.0212)</td>
            <td>.7454(.0209)</td>
            <td>.7519(.0341)</td>
            <td>.7965(.0225)</td>
            <td>.8089(.0176)</td>
        </tr>
        <td rowspan="6">Laplcian</td>
            <td rowspan="3">0.04</td>
            <td>RMSE</td>
            <td>.1113(.0085)</td>
            <td>.1110(.0084)</td>
            <td>.1117(.0085)</td>
            <td>.1125(.0083)</td>
            <td>.2642(.0135)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.6050(.0296)</td>
            <td>.5783(.0245)</td>
            <td>.5983(.0190)</td>
            <td>.6817(.0257)</td>
            <td>.7044(.0138)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.7719(.0212)</td>
            <td>.7482(.0199)</td>
            <td>.7688(.0161)</td>
            <td>.8184(.0137)</td>
            <td>.8329(.0083)</td>
        </tr>
        <tr>
            <td rowspan="3">0.06</td>
            <td>RMSE</td>
            <td>.2496(.0488)</td>
            <td>.2491(.0488)</td>
            <td>.2497(.0488)</td>
            <td>.2505(.0486)</td>
            <td>.1464(.0351)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.5700(.0427)</td>
            <td>.5967(.0316)</td>
            <td>.6083(.0578)</td>
            <td>.6783(.0187)</td>
            <td>.7050(.0265)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.7463(.0148)</td>
            <td>.7600(.0275)</td>
            <td>.7681(.0377)</td>
            <td>.8208(.0066)</td>
            <td>.8329(.0107)</td>
        </tr>
        <td rowspan="6">Salt and Pepper</td>
            <td rowspan="3">0.02</td>
            <td>RMSE</td>
            <td>.0859(.0005)</td>
            <td>.0856(.0003)</td>
            <td>.0864(.0004)</td>
            <td>.0823(.0003)</td>
            <td>.3253(.0037)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.5683(.0172)</td>
            <td>.5833(.0315)</td>
            <td>.5867(.0322)</td>
            <td>.6689(.0180)</td>
            <td>.7056(.0322)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.7463(.0148)</td>
            <td>.7427(.0163)</td>
            <td>.7521(.0230)</td>
            <td>.8116(.0050)</td>
            <td>.8394(.0134)</td>
        </tr>
        <tr>
            <td rowspan="3">0.1</td>
            <td>RMSE</td>
            <td>.1141(.0016)</td>
            <td>.1100(.0013)</td>
            <td>.1142(.0017)</td>
            <td>.0920(.0017)</td>
            <td>.2941(.0044)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.5178(.0434)</td>
            <td>.5356(.0306)</td>
            <td>.5033(.0487)</td>
            <td>.6306(.0288)</td>
            <td>.5850(.0257)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.7244(.0221)</td>
            <td>.7242(.0193)</td>
            <td>.7166(.0318)</td>
            <td>.8016(.0182)</td>
            <td>.7828(.0113)</td>
        </tr>
        <td rowspan="6">Block</td>
            <td rowspan="3">10</td>
            <td>RMSE</td>
            <td>.1064(.0007)</td>
            <td>.0989(.0005)</td>
            <td>.1056(.0007)</td>
            <td>.0828(.0003)</td>
            <td>.3276(.0030)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.4472(.0354)</td>
            <td>.4961(.0359)</td>
            <td>.4772(.0299)</td>
            <td>.6606(.0271)</td>
            <td>.6261(.0172)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.6381(.0283)</td>
            <td>.6744(.0323)</td>
            <td>.6673(.0299)</td>
            <td>.8116(.0132)</td>
            <td>.7721(.0061)</td>
        </tr>
        <tr>
            <td rowspan="3">15</td>
            <td>RMSE</td>
            <td>.1531(.0019)</td>
            <td>.1390(.0021)</td>
            <td>.1517(.0019)</td>
            <td>.1104(.0052)</td>
            <td>.3401(.0018)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.3633(.0161)</td>
            <td>.4150(.0511)</td>
            <td>.3656(.0349)</td>
            <td>.5783(.0282)</td>
            <td>.3028(.0228)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.5528(.0208)</td>
            <td>.6101(.0335)</td>
            <td>.5627(.0314)</td>
            <td>.7513(.0200)</td>
            <td>.4863(.0256)</td>
        </tr>
        <tr>
            <td rowspan="30">YaleB</td>
            <td rowspan="6">Uniform</td>
            <td rowspan="3">0.1</td>
            <td>RMSE</td>
            <td>.1232(.0005)</td>
            <td>.1227(.0004)</td>
            <td>.1235(.0005)</td>
            <td>.1235(.0004)</td>
            <td>.1044(.0003)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.1878(.0102)</td>
            <td>.1899(.0055)</td>
            <td>.1890(.0089)</td>
            <td>.1562(.0040)</td>
            <td>.1632(.0066)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.2674(.0154)</td>
            <td>.2586(.0124)</td>
            <td>.2599(.0135)</td>
            <td>.2399(.0136)</td>
            <td>.2064(.0137)</td>
        </tr>
        <tr>
            <td rowspan="3">0.3</td>
            <td>RMSE</td>
            <td>.3102(.0014)</td>
            <td>.3089(.0015)</td>
            <td>.3100(.0015)</td>
            <td>.3128(.0016)</td>
            <td>.2571(.0348)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.1641(.0307)</td>
            <td>.1819(.0265)</td>
            <td>.1706(.0300)</td>
            <td>.1316(.0086)</td>
            <td>.1327(.0097)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.2382(.0404)</td>
            <td>.2551(.0333)</td>
            <td>.2458(.0363)</td>
            <td>.1682(.0205)</td>
            <td>.1573(.0215)</td>
        </tr>
        <td rowspan="6">Gaussian</td>
            <td rowspan="3">0.05</td>
            <td>RMSE</td>
            <td>1.1221(.3938)</td>
            <td>1.1219(.3938)</td>
            <td>1.1221(.3938)</td>
            <td>1.1216(.3936)</td>
            <td>1.1160(.3902)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.1334(.0264)</td>
            <td>.1362(.0264)</td>
            <td>.1359(.0244)</td>
            <td>.1174(.0135)</td>
            <td>.1276(.0133)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.1922(.0511)</td>
            <td>.1865(.0492)</td>
            <td>.1840(.0581)</td>
            <td>.1357(.0344)</td>
            <td>.1416(.0134)</td>
        </tr>
        <tr>
            <td rowspan="3">0.08</td>
            <td>RMSE</td>
            <td>3.0621(.9219)</td>
            <td>3.0620(.9220)</td>
            <td>3.0621(.9219)</td>
            <td>3.0583(.9171)</td>
            <td>2.9515(.9138)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.0864(.0965)</td>
            <td>.0855(.0146)</td>
            <td>.0843(.0151)</td>
            <td>.0843(.0105)</td>
            <td>.0877(.0126)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.0965(.0396)</td>
            <td>.0925(.0338)</td>
            <td>.0956(.0361)</td>
            <td>.0775(.0146)</td>
            <td>.0794(.0192)</td>
        </tr>
        <td rowspan="6">Laplcian</td>
            <td rowspan="3">0.04</td>
            <td>RMSE</td>
            <td>1.6705(.6822)</td>
            <td>1.6703(.6822)</td>
            <td>1.6705(.6822)</td>
            <td>1.6692(.6817)</td>
            <td>1.6707(.6771)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.1208(.0261)</td>
            <td>.1197(.0262)</td>
            <td>.1188(.0294)</td>
            <td>.1017(.0169)</td>
            <td>.1166(.0123)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.1649(.0569)</td>
            <td>.1667(.0407)</td>
            <td>.1564(.0499)</td>
            <td>.1175(.0443)</td>
            <td>.1214(.0208)</td>
        </tr>
        <tr>
            <td rowspan="3">0.06</td>
            <td>RMSE</td>
            <td>4.3538(1.5452)</td>
            <td>4.3537(1.5452)</td>
            <td>4.3538(1.5452)</td>
            <td>4.3414(1.5343)</td>
            <td>4.2264(1.5424)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.0739(.0091)</td>
            <td>.0720(.0083)</td>
            <td>.0727(.0080)</td>
            <td>.0959(.0134)</td>
            <td>.0855(.0119)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.0658(.0259)</td>
            <td>.0638(.0263)</td>
            <td>.0602(.0174)</td>
            <td>.0988(.0181)</td>
            <td>.0764(.0193)</td>
        </tr>
        <td rowspan="6">Salt and Pepper</td>
            <td rowspan="3">0.02</td>
            <td>RMSE</td>
            <td>.0749(.0004)</td>
            <td>.0765(.0004)</td>
            <td>.0749(.0003)</td>
            <td>.0738(.0002)</td>
            <td>.1495(.0005)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.1903(.0091)</td>
            <td>.1852(.0106)</td>
            <td>.1959(.0139)</td>
            <td>.1575(.0055)</td>
            <td>.1730(.0070)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.2705(.0154)</td>
            <td>.2556(.0113)</td>
            <td>.2736(.0329)</td>
            <td>.2436(.0135)</td>
            <td>.2228(.0166)</td>
        </tr>
        <tr>
            <td rowspan="3">0.1</td>
            <td>RMSE</td>
            <td>.1213(.0020)</td>
            <td>.1100(.0009)</td>
            <td>.1214(.0018)</td>
            <td>.0779(.0026)</td>
            <td>.1467(.0238)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.1365(.0082)</td>
            <td>.1565(.0047)</td>
            <td>.1313(.0050)</td>
            <td>.1506(.0061)</td>
            <td>.1308(.0108)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.1971(.0151)</td>
            <td>.2115(.0163)</td>
            <td>.1750(.0114)</td>
            <td>.2217(.0101)</td>
            <td>.1586(.0106)</td>
        </tr>
        <td rowspan="6">Block</td>
            <td rowspan="3">10</td>
            <td>RMSE</td>
            <td>.1717(.0009)</td>
            <td>.1560(.0006)</td>
            <td>.1706(.0007)</td>
            <td>.1124(.0060)</td>
            <td>.1854(.0126)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.0994(.0072)</td>
            <td>.1123(.0118)</td>
            <td>.0917(.0035)</td>
            <td>.1210(.0113)</td>
            <td>.0941(.0079)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.1055(.0070)</td>
            <td>.1256(.0106)</td>
            <td>.0947(.0042)</td>
            <td>.1730(.0160)</td>
            <td>.0963(.0090)</td>
        </tr>
        <tr>
            <td rowspan="3">15</td>
            <td>RMSE</td>
            <td>.2669(.0013)</td>
            <td>.2594(.0014)</td>
            <td>.2664(.0013)</td>
            <td>.2540(.0010)</td>
            <td>.2542(.0069)</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>.0813(.0029)</td>
            <td>.0948(.0047)</td>
            <td>.0846(.0058)</td>
            <td>.0766(.0038)</td>
            <td>.0811(.0017)</td>
        </tr>
        <tr>
            <td>NMI</td>
            <td>.0748(.0080)</td>
            <td>.1068(.0133)</td>
            <td>.0845(.0170)</td>
            <td>.0731(.0079)</td>
            <td>.0747(.0021)</td>
        </tr>
    </tbody>
</table>

### 6.3. Reconstruction Effects
- Gaussian Noise Reconstruction
<p align="center">
  <img src="figures/yaleb-gau-recon.png">
  <br>
  Figure 12. Gaussian Noise Reconstruction (Noise Level: 0.16)
</p>

- Laplacian Noise Reconstruction
<p align="center">
  <img src="figures/yaleb-lap-recon.png">
  <br>
  Figure 13. Laplacian Noise Reconstruction (Noise Level: 0.1)
</p>

- Uniform Noise Reconstruction
<p align="center">
  <img src="figures/yaleb-uniform-recon.png">
  <br>
  Figure 14. Uniform Noise Reconstruction (Noise Level: 0.1)
</p>

- Block Occlusion Noise Reconstruction
<p align="center">
  <img src="figures/yaleb-block-10-recon.png">
  <br>
  Figure 15. Block Occlusion Noise Reconstruction (Block Size: 10)
</p>

<p align="center">
  <img src="figures/yaleb-block-20-recon.png">
  <br>
  Figure 16. Block Occlusion Noise Reconstruction (Block Size: 20)
</p>

- Salt and Pepper Noise Reconstruction
<p align="center">
  <img src="figures/l1normreg-yaleb-salt.png">
  <br>
  Figure 17. Salt and Pepper Noise Reconstruction (Noise Level: 0.1)
</p>

## 7. :sparkles: Project Structure
```
├── NumPyNMF/
│   ├── algorithm/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── nmf.py
│   │   ├── pipeline.py
│   │   ├── preprocess.py
│   │   ├── sample.py
│   │   ├── trainer.py
│   │   ├── user_evaluate.py
│   │   └── visualize.py
│   └── data/
│       ├── CroppedYaleB/
│       └── ORL/
├── figures/*.png
├── .gitignore
├── LICENSE
├── README.md
├── README.zh-CN.md
├── requirements.txt
├── run.ipynb
└── run.py
```

## 8. Update Log & TODO List
- 2023-10-20      
    - TODO List:
        - NumPy memory preallocation :x: (Conflicts with readability)
        - Reasons for algorithmic non-functionality (Done: 2023-11-24)
        - GUI interface (Done: 2024-02-01)
- 2023-11-10
    - Update Log:
        - Enhanced `algorithm.NMF` module
        - Fixed non-functionality of several algorithms
- 2023-11-24
    - Update Log:
        - Deprecated some modules and decoupled `algorithm.nmf`, which makes it flexible for users to transfer on other tasks
        - Integrated the basic functions in the deprecated modules into `BasicNMF`
- 2024-02-01
    - Update Log: 
        - Released user interface scripts.
        - Introduced advanced techniques.
        - Streamlined processes in a step-by-step manner
- 2024-02-07
    - Update log:
        - Constructed experiments utilizing multiprocessing techniques, which have resulted in significant acceleration
        - Added log recordings during experiments

## 9. :handshake: Contribution
We welcome contributions of any kind, whether it's suggesting new features, reporting bugs, or helping with code optimizations. Here are the steps to get started:

### 1. Fork the Project:
- Fork this repository by clicking the "Fork" button on the top right corner of this page.

### 2. Clone Your Fork:
```bash
git clone https://github.com/YOUR_USERNAME/PROJECT_NAME.git
```

Then navigate to the project directory:
```bash
cd PROJECT_NAME
```
### 3. Create a New Branch:
- Name your branch based on the change you're implementing, e.g., `feature/new-feature` or `bugfix/issue-name`:

```bash
git checkout -b branch-name
```

### 4. Commit Your Changes:
- Make your changes on this branch and then commit them.
  ```bash
  git add .
  git commit -m "Describe your changes here"
  ```
### 5. Push the Branch to Your Fork:
```bash
git push origin branch-name
```

### 6. Open a Pull Request:
- Go back to your fork on GitHub, and click "New Pull Request". Choose the branch you just pushed and submit the pull request.

### 7. Wait for Review:
- The maintainers of the project will review your pull request. They might request some changes or merge it.