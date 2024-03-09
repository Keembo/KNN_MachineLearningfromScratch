import numpy as np
from ml_from_scratch.neighbors import KNeighborsClassifier  # Import your class

# *******  Section 1: Load and Prepare Dataset *******
# Replace this with code to load your dataset
# Example using simplified data for demonstration
X_train = np.array([[1, 1], [2, 1.5], [1.5, 2], [4, 4], [3.5, 3]])
y_train = np.array([0, 0, 0, 1, 1]) 

# ******* Section 2: Create the KNN Model *******
knn_clf = KNeighborsClassifier(n_neighbors=3, p=2)  # Use Euclidean distance (p=2)

# ******* Section 3: Fit the Model *******
knn_clf.fit(X_train, y_train)

# ******* Section 4: Make Predictions *******
X_test = np.array([[2.5, 2.5], [3, 3.2]])
predictions = knn_clf.predict(X_test)
print(predictions)
