# Energy Consumption
<p align="center"><img width="1000" height="400" src="https://www.pexels.com/photo/540977/download/?search_query=&tracking_id=n0ja7c6z2v"></p>

Building energy consumption varies across a year, especially for countries in the temperate zones of the southern and northern hemispheres. With varying consumption comes varying demand, thus it becomes important for building managers to make better decisions to reasonably control all kinds of equipment. A well-managed, energy efficient building offers opportunities to reduce costs and reduce greenhouse gas emissions. However, as a result of randomness and noisy disturbance, it is not an easy task to realize accurate prediction of the building energy consumption. 

## Table of Contents
* **1. About the Project**
* **2. Getting Started**
* **3. Set up your environment**
* **4. Run Model Training**
* **4. Run Streamlit**

## Structuring a repository
To ensure reusability, a sensible repository structure is established as follows:

```bash
energy_consumption
├───ec
│   ├───.ipynb_checkpoints
│   ├───analysis
│   │   ├─── __init__.py
│   │   ├─── clustering.py
│   │   ├─── feature_engineering.py
│   │   └─── impute.py
│   ├───base
│   │   ├─── __init__.py
│   │   └─── logger.py
│   ├───data
│   │   ├───analysis
│   │   ├───forecast
│   │   │   ├───consumption
│   │   │   └───temperature
│   │   └───train
│   ├───output
│   ├───train
│   │   ├─── __init__.py
│   │   ├─── modelling.py
│   │   └───__pycache__
├───logs
├── .gitignore
├── README.md
├── environment.yml
└── requirements.txt

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
    - <b><u>Linear Regression</u></b>
    - <b><u>Random Forest</u></b>
    - <b><u>Light Gradient Boosting Method</u></b>

### Key Takeaways

How is this project different from the rest? Here are some pain points to give you a general view of the focus areas of the project: 
- There is a large number of buildings, in fact, 758 in total! 
- The buildings come from different locations, and these locations are unknown 
- There is a lot of missing data in the dataset
- Timestamp of all buildings are scattered

## 2. Getting Started

To get started, you may clone the repository locally as follows:
  
In your terminal, use `git` to clone the repository locally.

```bash
https://github.com/gracengu/building_energy_consumption_forecast.git
```
  
Alternatively, if you don't have experience with git, you may download the zip file of the repository at the top of the main page of the repository. However, it is most recommended to use git for version control purpose. 
    

## 3. Set up virtual environment

Setting up virtual environment avoids conflict in dependencies: 


- For `pip` users

To set up **virtual environment**, use the `virtualenv` command. 

```bash
pip install virtualenv
virtualenv python3.7_energy
```

**Note:** When creating a virtual environment, best practice is to specify the python version as part of the name of the environment for future reference of other developers.  

To **activate the environment**, run the activate script. 

```bash
python3.7_energy\Script\activate
```

To **install the requirements**

```bash
pip install -r requirements.txt
```

## 4. Run Model Training

The models are not uploaded in github. After setting up the repository, you may run the model training as follows:

```python
  cd src
  python training_modules.py
```

## 5. Run Streamlit

For this project, all results are demonstrated via streamlit. You may run the streamlit as follows: 

```python
  streamlit run app.py
```
