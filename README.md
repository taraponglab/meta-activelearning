# **Active Stack-Deep Learning with Strategic Sampling for Small and Imbalance Chemical Toxicity Prediciton**

![Sample Figure](graphic_abstract.png)

### Darlene Nabila Zetta†, Watshara Shoombuatong‡, and Tarapong Srisongkram*

†Graduate School in the Program of Pharmaceutical Sciences, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen, 40002, Thailand. (darlenenabilazetta.d@kkumail.com)

‡Center for Research Innovation and Biomedical Informatics, Faculty of Medical Technology, Mahidol University, Bangkok, 10700, Thailand. (watshara.sho@mahidol.ac.th)

*Division of Pharmaceutical Chemistry, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen, 40002, Thailand. (tarasri@kku.ac.th)

Full paper submitted in **ACS Omega**.

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Features Extraction](#features-extraction)
- [Training and Evaluate the Model](#training-the-model)
- [Reproducing Results](#reproducing-results)
- [MIT License](#mit-license)

### 📖 Overview
This repository implements Active Stack-Deep Learning with Strategic Sampling for Small and Imbalance Chemical Toxicity Prediction. The pipeline includes:
- Data preprocessing
- Feature extraction
- Model training
- Performance evaluation

### 🧰 Requirements
- Python >= 3.8 or above
- Install dependencies from `requirements.txt`:
  ```bash
  pip install -r requirements.txt
- Install the required packages:
```pip install -r requirements.txt```

### 📂 Data Preparation
The `data/` directory is organized as follows:

```
data/
├── train/ # Training data
├── test/ # Testing data
├── subsets/ # Selected initial subsets
└── pool/ # Remaining unlabeled compounds
```

To preprocess the data, run:

```bash
python preprocess.py
```
### 🔑 Features Extraction
The features extraction of twelve fingerprints calculated with python file: The extraction of twelve molecular fingerprints is performed using the following script:

python calculate_fp.py

This script is supported by the `fingerprints_xml/` folder, which contains the necessary fingerprint definitions.

### 📈 Training and Evaluate the Model
The training and evaluation process using the processed data includes the following steps:

1. Divide the subset data for sampling
Run:
```
python divide_sampling.py
```
This will generate multiple k-ratio subset samplings saved in one folder.

2. Train models on subset samplings
Run:
```
python train_meta_sampling.py
```
This script trains models on each subset sampling.

3. Train the stacking ensemble and evaluate
Run:
```
python train_average_probability.py
```
This trains a CNN-based stacking ensemble on the subsets. The average predictions are calculated and the evaluation results are saved as a CSV file.

4. Apply active learning selection strategies
You can run one of the following, depending on the desired strategy:
```
python entropy_cal.py
python margin_cal.py
python uncertainty_cal.py
```
These scripts select new compounds from the pool based on entropy, margin, or uncertainty, and generate updated subset and pool files.

5. Split the pool dataset
Run:
```
python pool_split.py
```
This script separates the remaining pool data into a new folder for the next iteration.

Repeat the steps above for each active learning iteration until the desired number of compounds or performance is achieved.

### 🔁 Reproducing Results
To reproduce the results reported in the paper:

1. Follow the requirements and data preprocessing steps.

2. Run the training and evaluation scripts in sequence as described above.

3. The outputs and evaluation results will be saved in the specified folders.

### MIT License
Copyright (c) [2025] [Dr.Tarapong Srisongram]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
