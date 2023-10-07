# predicting fault-revealing mutants by estimating the difficulty of killing them

 **Note**
To potential supervisors, committee members, hiring managers, and others who are reviewing this repository to assess my implementation skills, I highly recommend visiting my website. In the 'NOTES' section, you can find a dedicated section titled 'Potential Questions,' where I have provided the necessary context for a more comprehensive assessment. [MyWebsite](https://taharostami.github.io/notes/)

## Introduction

The source code for the paper: FrMi: Fault-revealing Mutant Identification using killability severity [(link)](https://www.sciencedirect.com/science/article/abs/pii/S0950584923001623)

## Data

Thanks to Titcheu Chekam et al., the original datasets are available at   https://mutationtesting.uni.lu/farm/.

In this repository, I provided the preprocessed version of the datasets in which I added the proposed feature. These datasets are available in the folder named *data*.

## Results
I also provided the main results of the study in *results/main_results.parquet*. This file could be used to validate the study's findings, analyse the results, etc.

## Code
The *src/train.py* is a minimum and straightforward implementation of the proposed method. Please extract the datasets provided in the *data* folder for executing the code. Also, note that the datasets used in this study require a large memory volume. Please split the code into two parts if you would like to replicate the study but with less memory usage. In the first part, train regression models for all ten folds. Then in the second part, train the classifiers.

The file *src/eval.py* could be used for reporting the results, e.g., *results/main_results.parquet*.

 
