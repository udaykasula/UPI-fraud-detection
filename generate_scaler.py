from sklearn.preprocessing import StandardScaler
import pickle

# Example data to fit the scaler (replace this with your actual data)
data = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(data)

# Save the scaler to a file
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("scaler.pkl has been successfully generated.")