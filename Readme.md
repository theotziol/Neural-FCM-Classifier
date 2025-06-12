# Neural-FCM

## Table of Contents

- [Abstract](#Abstract)
- [Installation](#Instalation)
- [Usage](#Usage)
- [Examples](#Examples)
- [License](#license)

## Abstarct

Neural-FCM is a novel learning framework that integrates deep learning concepts into Fuzzy Cognitive Map (FCM) modeling to improve classification performance. By employing a hybrid artificial neural network (ANN) decoder architecture, Neural-FCM transforms input data instances into FCM weight matrices. The method leverages deep learning optimization techniques, incorporating FCM reasoning into the training process by embedding it within a differentiable loss function. This approach ensures that the network outputs weight matrices that enhance FCM inference accuracy, while maintaining the fundamental interpretability of FCMs. Neural-FCM's dynamic weight generation allows for instance-specific matrix outputs, similar to Fuzzy Grey Cognitive Maps, enabling the model to adapt effectively to input data and deliver robust classification results. The use of high-level tools like Python and TensorFlow supports the practicality and reproducibility of the method, fostering further research and application in various domains. 

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

This repository contains the following folders: 

1. In *data* folder there should be placed the datasets in .csv format and into subfolders to be consistent with how they are imported into code within *main.py*. The employed datasets, besides CAD, can be found online and are properly cited in our paper.
2. The *experiments* folder is used for storing the results of experiments as subfolders using the time and date of the experiment.

 
## Code explanation

Information about python scripts:

1. [main.py](main.py): Is the main script that handles the experiments. It imports the Neural-FCM algorithm along with all the other necessary libraries for preprocessing, learning, post-processing and results storing.
2. [dynamic_fcm.py](dynamic_fcm.py): Contains the code for Neural-FCM. It handles the tf model creation, the loss function initialization and FCM inference. 
3. [preprocessing.py](preprocessing.py): Contains the code for data preparation. 
4. [fcm_libr_tensorflow.py](fcm_libr_tensorflow.py): Contains the tensorflow code for FCM supportive functions such as sigmoid.

## License
This project is going to be licensed under the [MIT License](LICENSE.txt) after acceptance for publication. 


For more information you should contact the corresponding author/developer [**Theodoros Tziolas**](https://scholar.google.gr/citations?user=ww_3OmIAAAAJ&hl=el). 
Email: <ttziolas@uth.gr>