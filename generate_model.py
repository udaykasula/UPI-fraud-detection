from sklearn.ensemble import RandomForestClassifier
import pickle

# Example training data (replace with your actual data)
X_train = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
y_train = [0, 1, 0]  # Labels (0 = Not Fraud, 1 = Fraud)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model.pkl has been successfully generated.")