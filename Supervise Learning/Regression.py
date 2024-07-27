import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets,linear_model

dataset = datasets.fetch_california_housing()
# print(dataset.keys())
# print(dataset['DESCR'])

y = dataset['target']

X= dataset['data'][:,[2]]

# plt.scatter(X,y)
# plt.xlabel('AveRooms')
# plt.ylabel('Price')
# plt.show()
model = linear_model.LinearRegression()

model.fit(X,y)

print('Price','=',model.coef_,'* AveRooms','+',model.intercept_)
y_pred = model.coef_ * X + model.intercept_

plt.scatter(X,y,color = 'black')
plt.scatter(X,y_pred,color ="blue",linewidths=3)
plt.xlabel('AveRooms')
plt.ylabel('Price')
# plt.show()

# print(model.coef_ *7 + model.intercept_)
print(model.predict([[7]]))