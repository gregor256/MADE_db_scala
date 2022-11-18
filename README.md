Linear regression on scala
---
## Usage:
Install dependencies than are contained in `build.sbt` <br> 
Run  `src/main/scala/Main.scala` in any way. <br>
It will fit linear regression model on data containing in `data/train_data.csv`.
Columns (0, 1, 2, 3) in that file is X, column 4 is Y.
Then model will predict target value y on training dataset and on test dataset(`data/train_data.csv`). 
Prediction results are stored in `predictions/y_train_predicted.csv` and 
`predictions/y_test_predicted.csv` respectively.
Evaluation of model quality will be stored at `evaluation/model_evaluation.txt`.

## Data creation.
```pip install -r requirements.txt``` <br>

Dataframes are created by calling 
```python gen_data.py N```

where 0.66*N and 0.33*N are amount of objects in train and test datasets.
It creates two dataframes: train and test and store them in folder `data`.  
4th column is linear combination of (1, 2, 3, 4) with random noise. 

### Example:
```python gen_data.py 800```


