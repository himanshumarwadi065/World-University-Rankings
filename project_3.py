import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

data = pd.read_csv("World University Rankings.csv" , encoding="latin1")
df = pd.DataFrame(data)
df = df.fillna(0)
df = df.fillna("unknown")

# print(df.columns)

# column_axis = np.array([df[["scores_teaching" , "scores_international_outlook"]]]).reshape(-1,1)
# random_col = np.array([df["scores_citations"]])
column_axis = df[["scores_teaching", "scores_international_outlook"]].values
random_col = df["scores_citations"].values
model = KNeighborsRegressor(n_neighbors=50)
model.fit(column_axis,random_col)

citations_pre = pd.DataFrame([[99.99 , 99 ]],columns=["scores_teaching","scores_international_outlook"])

cit_pre  = model.predict(citations_pre)

print("Predict is Citations Score's : ",cit_pre[0])
