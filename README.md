<p align="center">
  <!-- <a href="https://github.com/RikilG/Machine-Learning">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->
  <h2 align="center">Machine Learning</h2>
  <p align="center">
    Implementations of popular machine learning models
  </p>
</p>


## Table of Contents

- [Table of Contents](#table-of-contents)
- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
- [Project Layout](#project-layout)
- [License](#license)
- [Project Contributors](#project-contributors)


## About The Project

This repo contains implementations of popular machine learning models. This implementations
are part of our course work in Machine Learning.


## Getting Started

To get a local copy up and running follow these simple steps.  
Python version >= 3.6 is required to run this algorithms without any problems.


### Dependencies

This is an example list of dependencies present/used in the project
 - [numpy](https://www.numpy.org)
 - [scipy](https://www.scipy.org)
 - [pandas](https://www.pandas.pydata.org)
 - [matplotlib](https://www.matplotlib.org)
 - [nltk](https://www.nltk.org) (used in naives bayes classifier for stemming and tokenizing words in database)
 - [tqdm](https://github.com/tqdm/tqdm) (used for progressbars in neural net)


### Installation
 
1. Clone the Machine-Learning
```sh
git clone https://github.com/RikilG/Machine-Learning.git
cd Machine-Learning
```
2. Install python modules
If you have anaconda installed, run this command to fetch all packages
```sh
conda install scikit-learn numpy pandas tqdm matplotlib nltk
```
Else, install all the required packages using pip
```sh
pip -r requirements.txt
```
3. Run the files in src folder with `python <filename.py>` command


## Project Layout
```
repo root directory
├── AssignmentPDFs
│   ├── Assignment 1 - Machine Learning.pdf
│   └── Assignment 2 - Machine Learning.pdf
├── datasets # Datasets used by all algorithms
│   ├── a1_d1.csv
│   ├── ...
│   └── xorgate.csv
├── Images # Neural net images
│   ├── 2Layer0.2Alpha.png
│   ├── ...
│   └── Screenshot_20200426_100220.png
├── Reports
│   ├── BITS F464 - Report 1.pdf
│   └── BITS F464 - Report 2.pdf
├── src
│   ├── Fischers_Discriminant
│   │   └── fisher_discr.py
│   ├── Logistic_Regression
│   │   └── LogisticRegression.py
│   ├── Naive_Bayes
│   │   ├── naive_bayes.py
│   │   └── preprocess.py
│   └── Neural_Network
│       ├── neural_network.py
│       └── nn_core.py
├── LICENSE
├── README.md
├── requirements.txt
└── tree.txt
```


## License

Distributed under the MIT License. See `LICENSE` for more information.


## Project Contributors

- Rikil Gajarla          - 2017A7PS0202H
- L Srihari              - 2017A7PS1670H
- Raj Kashyap Mallala    - 2017A7PS0025H

Project Link: [https://github.com/RikilG/Machine-Learning](https://github.com/RikilG/Machine-Learning)