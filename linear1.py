from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np

data = datasets.load_diabetes()
model = linear_model.LinearRegression()

X = data.data[:, np.newaxis, 3]

X_train = X[:-20]
X_test = X[-20:]

y_train = data.target[:-20]
y_test = data.target[-20:]

model.fit(X_train, y_train)

print('Koeffizient : ', model.coef_)
print("mittlere quadratische Fehler: %.2f" % np.mean((model.predict(X_test) - y_test) ** 2))
print('Ergebnis: %.2f' % model.score(X_test, y_test))

# Plot
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, model.predict(X_test), color='green', linewidth=2)

plt.xticks(())
plt.yticks(())

plt.show()