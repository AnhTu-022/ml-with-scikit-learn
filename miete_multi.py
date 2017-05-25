from sklearn.linear_model import LinearRegression

X = [[60, 2], [70, 1], [85, 3], [110, 4], [115, 1], [130, 2]] 
y = [[550], [750], [850], [950], [1350], [1400]]

model = LinearRegression()
model.fit(X,y)

X_test1 = [120, 1]
X_test4 = [120, 4]

print ("Zustand Erstbezug - Preis : € %.2f" % model.predict([X_test1])[0])
print ("Zustand vereinbar - Preis : € %.2f" % model.predict([X_test4])[0])
#print (lstsq(X, y)[0])
