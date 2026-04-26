import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. DATABASE CONNECTION
# ==========================================
# Replace 'password' and 'your_db_name' with your actual Postgres credentials
engine = create_engine('postgresql://postgres:1234@localhost:5432/customerChurn')

print("--- Step 1: Loading Data from PostgreSQL ---")
df = pd.read_sql_query('SELECT * FROM "telco_cleaned"', engine)

# ==========================================
# 2. DATA PRE-PROCESSING (Encoding)
# ==========================================
print("--- Step 2: Encoding Categorical Data ---")
# This converts text (Yes/No, Contract types) into 0s and 1s
df_encoded = pd.get_dummies(df, columns=[
    'Gender', 'Internet Service', 'Contract', 
    'Payment Method', 'Senior Citizen'
])

# Remove columns that are not useful for prediction (IDs and text reasons)
cols_to_drop = ['CustomerID', 'Churn Reason', 'Churn Value', 'Churn Score']
numeric_df = df_encoded.select_dtypes(include=['number', 'bool'])

# ==========================================
# 3. CORRELATION HEATMAP
# ==========================================
print("--- Step 3: Generating Correlation Heatmap ---")
plt.figure(figsize=(14, 8))
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# ==========================================
# 4. MACHINE LEARNING (Random Forest)
# ==========================================
print("--- Step 4: Training Random Forest Model ---")
X = numeric_df.drop(['Churn Value', 'Churn Score'], axis=1, errors='ignore')
y = numeric_df['Churn Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 5. RESULTS & BUSINESS INSIGHTS
# ==========================================
print("\n--- MODEL PERFORMANCE REPORT ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot Feature Importance
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title('Top 10 Business Drivers of Churn')
plt.xlabel('Importance Score (How much this variable affects Churn)')
plt.show()