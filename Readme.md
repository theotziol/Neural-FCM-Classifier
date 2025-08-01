# Neural-FCM: A Deep Learning Approach for Weight Matrix Optimization in Fuzzy Cognitive Map Classifiers

Neural-FCM has been accepted for publication in [Applied Intelligence](https://link.springer.com/journal/10489). 

## Table of Contents

- [Abstract](#abstract)
- [Installation](#instalation)
- [Usage](#usage)
- [License](#license)


## Abstract

The demand for interpretable and accurate machine learning models continues to grow, especially in critical domains. The data-driven Fuzzy Cognitive Map (FCM) classifier is an interpretable and transparent decision-making method. Its core element, the weight matrix, is derived using predominantly population-based supervised learning methods which often suffer from degraded performance. Recent research has adopted gradient-based learning techniques to compete with the predictive performance of black-box models. Nonetheless, such methods modify foundational principles and compromise interpretability, highlighting the necessity to improve existing approaches. In this work, we introduce a novel learning and structural modeling method, termed Neural-FCM, which leverages deep neural networks and gradient descent to enhance the accuracy and robustness of FCM learning. Neural-FCM employs a hybrid network comprising both dense and convolutional layers and is trained using a categorical cross-entropy loss function specifically aligned with FCM reasoning. This hybrid model is trained to output instance-specific weight matrices for effective and targeted FCM inference, introducing structural adaptability, a feature not supported by previous static or globally optimized approaches. Focusing on generalization across domains, the Neural-FCM approach is evaluated on different classification tasks across six widely used public datasets and one proprietary medical dataset, consistently showing improved predictive performance. Notably, the comparative analysis against standard population-based FCM learning methods reveals consistent accuracy improvements, with gains of up to 34%. While less transparent gradient-based methods also yield improved accuracy, Neural-FCM demonstrates competitive or superior performance in most cases, with accuracy improvements ranging from 1% to 6% across different domains, while preserving the underlying interpretability. The performance enhancement and the use of instance-specific matrices contribute to the broader goal of developing gradient-based models that balance computational efficiency with the intrinsic FCM interpretability.

## Instalation

1. Clone the repository:

```bash
 git clone https://github.com/theotziol/neural-fcm-classifier.git
```

2. Download as a zip file

3. To use this project you must install the packages of **requirements.txt** file and having a **python 3.10 version (recommended)** installed on your machine. Once python and pip are installed run:

```bash
pip install -r requirements.txt
```

## Usage

### Folders

This repository contains the following folders:

1. In *data* folder there should be placed the datasets in .csv format and into subfolders to be consistent with how they are imported into code within [main.py](main.py) script. The employed datasets, besides CAD, can be found online and are properly cited in our paper.
2. The *experiments* folder is used for storing the results as subfolders using the current datetime as folder's name.

### Code explanation

Information about python scripts:

1. [main.py](main.py): Is the main script that handles the experiments. It imports the Neural-FCM algorithm along with all the other necessary libraries for preprocessing, learning, post-processing and results storing.
2. [dynamic_fcm.py](dynamic_fcm.py): Contains the code for Neural-FCM. It handles the tf model creation, the loss function initialization and FCM inference.
3. [preprocessing.py](preprocessing.py): Contains the code for data preparation.
4. [fcm_libr_tensorflow.py](fcm_libr_tensorflow.py): Contains the tensorflow code for FCM supportive functions such as sigmoid.

## License

This project is licensed under the [MIT License](LICENSE.txt).

For more information you should contact the corresponding author/developer [**Theodoros Tziolas**](https://scholar.google.gr/citations?user=ww_3OmIAAAAJ&hl=el).
Email: <ttziolas@uth.gr>
