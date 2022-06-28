import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from sklearn import linear_model



data = pd.read_csv(r'C:\Users\ibrahima\Desktop\syra.txt')
data.head()

X = data[["number","max","decente"]]
Y = data[["montee"]]


# Generating training and testing data from our data:
# We are using 80% data for training.
train = data
#[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.8))):]

#Modeling:
#Using sklearn package to model data :
regr = linear_model.LinearRegression()

train_x = np.array(train[["number","max","decente"]])
train_y = np.array(train["montee"])

regr.fit(train_x,train_y)

test_x = np.array(test[["number","max","decente"]])
test_y = np.array(test["montee"])


# print the coefficient values:
coeff_data = pd.DataFrame(regr.coef_ , X.columns , columns=["Coefficients"])

print(coeff_data)


#Now let’s do prediction of data:
Y_pred = regr.predict(test_x)
# Check accuracy:
from sklearn.metrics import r2_score
R = r2_score(test_y , Y_pred)
print ("R² :",R)

for i in range(5):
    print('test  ', test_y[i], '\t\t   pred  ', Y_pred[i])


data5 = data[["number","montee"]]
plt.scatter(data5["number"],data5["montee"],color="blue")


data1 = data[["max","montee"]]
plt.scatter(data1["max"],data1["montee"],color="red")


data2 = data[["decente","montee"]]
plt.scatter(data2["decente"],data2["montee"],color="green")

plt.legend(['number','max','descente'])


plt.show()
plt1.scatter(data5["number"],data5["montee"],color="red")
plt1.show()


