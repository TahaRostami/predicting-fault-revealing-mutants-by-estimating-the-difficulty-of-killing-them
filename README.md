# predicting fault-revealing mutants by estimating the difficulty of killing them

## Data

Thanks to Titcheu Chekam et al., the original datasets are available at   https://mutationtesting.uni.lu/farm/.

In this repository, I provided the preprocessed version of the datasets in which I added the proposed feature. These datasets are available in the folder named *data*.

## Results
I also provided the main results of the study in *main_results.parquet*. This file could be used to validate the study's findings, analyse the results, etc.

## Code
The *train.py* is a minimum and straightforward implementation of the proposed method. Please extract the datasets provided in the *data* folder for executing the code. Also, note that the datasets used in this study require a large memory volume. Please split the code into two parts if you would like to replicate the study but with less memory usage. In the first part, train regression models for all ten folds. Then in the second part, train the classifiers.

The file *src/eval.py* could be used for reporting the results, e.g., *main_results.parquet*.

 
