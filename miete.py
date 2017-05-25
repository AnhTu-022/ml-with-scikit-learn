import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

X = [[60], [70], [85], [110], [115], [130]] 
y = [[550], [650], [900], [1050], [1250], [1400]]
plt.figure()
plt.title('Miete') 
plt.xlabel('Wohnungsgroeße')
plt.ylabel('Miethoehe')
plt.plot(X, y, 'k.')
plt.axis([0, 150, 0, 1500])
plt.grid(True) 
#plt.show()

model = LinearRegression() 
model.fit(X, y)
print ("Eine 140 qm Wohnung hat eine Miethöhe von %.2f" % model.predict([45])[0])

# Plot outputs
plt.scatter(X, y,  color='black')
plt.plot(X, model.predict(X), color='blue',linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()