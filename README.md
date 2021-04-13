# Energy Consumption
<p align="center"><img width="1000" height="400" src="https://www.pexels.com/photo/540977/download/?search_query=&tracking_id=n0ja7c6z2v"></p>

Building energy consumption varies across a year, especially for countries in the temperate zones of the southern and northern hemispheres. With varying consumption comes varying demand, thus it becomes important for building managers to make better decisions to reasonably control all kinds of equipment. A well-managed, energy efficient building offers opportunities to reduce costs and reduce greenhouse gas emissions. However, as a result of randomness and noisy disturbance, it is not an easy task to realize accurate prediction of the building energy consumption. 

## Table of Contents
* **1. About the Project**
* **2. Getting Started**
* **3. Set up your environment**
* **4. Open your Jupyter notebook**


## Structuring a repository
To ensure reusability, a sensible repository structure is established as follows:

```bash
energy_consumption
├── docs
│   ├── make.bat
│   ├── Makefile
│   └── source
│       ├── conf.py
│       └── index.rst
├── src
│   └── analysis
│       └── __init__.py
│       └── processing.py
|       └── feature_engineer.py
|       └── statistical_analysis.py
|   └── Config.py
├── .gitignore
├── README.md
├── environment.yml
├── requirements.txt

```

## 1. About the Project

### Objective

The objective of this project is to forecast energy consumption based on temperature and other building information.
Three (3) time horizons for predictions are defined:
- Forecasting the consumption for each hour in the next day (24 predictions).
- Forecasting the consumption for each day in the coming week (7 predictions).
- Forecasting the consumption for each week in the coming two weeks (2 predictions).

### Features

The following building attributes are provided: 
- Temperature
- Base Temperature
- Surface Area
- Monday is Day Off, Tuesday is Day Off, Wednesday is Day Off, Thursday is Day Off, Friday is Day Off, Saturday is Day Off and Sunday is Day Off
### Key Concepts

The project revolves mainly about the following key data science approaches:
  - <b><u>Missing data imputation </b></u>
  - <b><u>Building Clusterings (Unsupervised Learning)</u></b>
  - <b><u>Time Series Analysis & Forecasting</u></b>
  - <b><u>Long Short-Term Memory (LSTM)</u></b>

### Key Takeaways

How is this project different from the rest? Here are some pain points to give you a general view of the focus areas of the project: 
- There is a large number of buildings, in fact, 758 in total! 
- The buildings come from different locations, and these locations are unknown 
- There is a lot of missing data in the dataset
- Timestamp of all buildings are scattered



## 2. Getting Started
- If you prefer to use the `conda` package manager (which ships with the Anaconda distribution of Python), you may clone the repository locally as follows:
  
    In your terminal, use `git` to clone the repository locally.
    
    ```bash
    https://github.com/gracengu/building_energy_consumption_forecast.git
    ```
    
    Alternatively, if you don't have experience with git, you may download the zip file of the repository at the top of the main page of the repository. However, it is most recommended to use git for version control purpose. 
    
- Prefer to use `pipenv`, which is a package authored by Kenneth Reitz for package management with `pip` and `virtualenv`, or

## 3. Set up virtual environment

Setting up virtual environment avoids conflict in dependencies: 

- For  `conda` users

If this is the first time you're setting up your compute environment, 
use the `conda` package manager 
to **install all the necessary packages** 
from the provided `energy_consumption.yml` file.

```bash
conda env create -f environment.yml
```

To **activate the environment**, use the `conda activate` command.

```bash
conda activate customer_segmentation
```

To **update the environment** based on the `environment.yml` specification file, use the `conda update` command.

```bash
conda env update -f environment.yml
```

- For `pip` users

Please install all of the packages listed in the `requirement.txt`. 
An example command would be:

```bash
pip install -r requirement.txt
```


## 4. Open your Jupyter notebook

1. You will have to install a new IPython kernelspec if you created a new conda environment with `environment.yml`.
    
    ```python
    python -m ipykernel install --user --name customer_segmentation --display-name "customer_segmentation"
    ```

You can change the `--display-name` to anything you want, though if you leave it out, the kernel's display name will default to the value passed to the `--name` flag.

2. In the terminal, execute `jupyter notebook`.

Navigate to the notebooks directory and open notebook:
  - ETL: `Analysis.ipynb` [Placeholder]
  - Modelling: `Train.ipynb` [Placeholder]
