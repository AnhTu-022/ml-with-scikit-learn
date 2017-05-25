from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt

y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] 
y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1] 
confusion_matrix = confusion_matrix(y_test, y_pred) 

print(confusion_matrix) 
plt.matshow(confusion_matrix) 
plt.colorbar()

plt.title('Konfusionsmatrix') 
plt.ylabel('Richig label')
plt.xlabel('Vorhersagen label') 
plt.show()