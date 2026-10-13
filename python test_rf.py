from sklearn.ensemble import RandomForestClassifier
import numpy as np

print("Testing Random Forest...")

X = np.random.rand(1000, 50)
y = np.random.randint(0, 7, 1000)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X, y)

print("RF WORKS")