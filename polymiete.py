from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt 


X = [[60], [70], [85], [110], [115], [130], [30], [50], [100]] 
y = [[550], [600], [700], [1150], [1350], [1400], [700], [500], [1000]]

X_test = [[60], [85], [110], [130]]
y_test = [[550], [850], [950], [1400]]

model = LinearRegression()
model.fit(X,y)

regressor = LinearRegression() 
regressor.fit(X, y)
xx = np.linspace(0, 200, 500)
yy = regressor.predict(xx.reshape(xx.shape[0], 1)) 
plt.plot(xx, yy)

poly_features = PolynomialFeatures(degree=3) 

X_train_poly = poly_features.fit_transform(X) 
X_test_poly = poly_features.transform(X_test)
regressor_poly = LinearRegression() 

regressor_poly.fit(X_train_poly, y)
xx_poly = poly_features.transform(xx.reshape(xx. shape[0], 1))
plt.plot(xx, regressor_poly.predict(xx_poly), c='r', linestyle='--')

plt.title('Miethöhe') 
plt.xlabel('Wohnungsgröße m2') 
plt.ylabel('Miete €') 

plt.axis([0, 150, -1000, 2500])
plt.grid(True)
plt.scatter(X, y) 

plt.show()

plot_learning_curves(regressor_poly,X,y)

print ('Simple linear regression r-squared %.4f' %  regressor.score(X_test, y_test))
print ('Quadratic regression r-squared %.4f' % regressor_poly.score(X_test_poly, y_test))