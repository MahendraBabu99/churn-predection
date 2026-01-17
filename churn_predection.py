import sklearn 
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\chapa mahindra\Downloads\archive (1)\Churn.csv")
df = df.drop('customerID', axis=1)
X = df.drop('Churn', axis=1)
y = df['Churn']

num_cols = X.select_dtypes(include = ['int64','float64']).columns
cat_cols = X.select_dtypes(include = ['object', 'category']).columns

preprocessing = ColumnTransformer(
    transformers = [
        ('num' , SimpleImputer(strategy="mean"), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('encoder', OneHotEncoder(handle_unknown="ignore"))
        ]),cat_cols)
    ])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=45)
model = Pipeline([
    ('preprocessing', preprocessing),
    ('classifier', RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=1))
])

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", roc_auc)
