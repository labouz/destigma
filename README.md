# Destigmatization Framework

This repository contains the code and notebooks for our study on reducing stigma in online conversations about substance use disorders (SUD) using large language models (LLMs). The goal is to transform stigmatizing language into more empathetic expressions to foster a more supportive digital environment for individuals affected by SUD.

## Repository Structure

- `task1.py`: This script is responsible for the initial filtering and identification of posts related to substance use from a large dataset of Reddit posts.
- `task2.py`: This script focuses on detecting stigmatizing language within the filtered posts identified in Task 1. It uses predefined criteria to label posts that contain harmful stereotypes or language.
- `task3.py`: This script handles the de-stigmatization of the identified posts using large language models. It transforms the stigmatizing language into more empathetic expressions.
- `task3_C.py`: An extension of `task3.py`, this script incorporates stylistic profiling to ensure that the de-stigmatized language maintains the original post's emotional tone and stylistic features.
- `data_analysis.ipynb`: A Jupyter notebook that provides a detailed analysis of the collected data, including the distribution of stigmatizing language, types of substances mentioned, and other relevant findings (e.g. LIWC).
- `stigma_eda.ipynb`: An exploratory data analysis (EDA) notebook that investigates the characteristics and patterns of stigmatizing language in the dataset. It includes visualizations and preliminary findings that inform the development of the de-stigmatization models.

## Data and Metadata

- **Metadata**: Metadata will be added to the repository to provide additional context and information about the dataset.
- **Full Data**: The full dataset can be provided upon request. Please contact the repository maintainers for access.

## Paper

## Contact
For questions regarding paper or data, please contact [lb3338] at [drexel.edu]
