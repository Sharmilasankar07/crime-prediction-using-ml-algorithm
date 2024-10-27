# predicting_crime.py

# 1) IMPORT LIBRARIES

# Computation and Structuring:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Modeling:
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Testing:
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Import the visualization function
from visualize import plot_confusion_matrix

#--------------------------------------------------#

# 2) DATA IMPORT AND PRE-PROCESSING

# Import full data set
df = pd.read_csv('MCI_2014_to_2017.csv', sep=',')

# List of relevant columns for model
col_list = [
    'OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOY', 'OCC_DOW', 'OCC_HOUR',
    'MCI_CATEGORY', 'DIVISION', 'HOOD_158', 'PREMISES_TYPE'
]

# Dataframe created from list of relevant columns
df2 = df[col_list]
df2 = df2[df2['OCC_YEAR'] > 2013]  # Ignore old crimes before 2014

# Factorize dependent variable column:
crime_var = pd.factorize(df2['MCI_CATEGORY'])  # Convert crimes to integer values
df2['MCI_CATEGORY'] = crime_var[0]
definition_list_MCI = crime_var[1]  # Create index reference for the crime categories

# Factorize all categorical independent variables:

# Factorize PREMIS_TYPE
df2['PREMISES_TYPE'] = pd.factorize(df2['PREMISES_TYPE'])[0]

# Factorize all other independent variables
factorize_columns = ['OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOY', 'OCC_DOW', 'OCC_HOUR', 'DIVISION', 'HOOD_158']

for col in factorize_columns:
    df2[col] = pd.factorize(df2[col])[0]

# Set X and Y:
X = df2.drop(['MCI_CATEGORY'], axis=1)  # X without the target variable
y = df2['MCI_CATEGORY']  # y as the target variable

# Split the data into train and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

#--------------------------------------------------#

# 3) MODELING AND TESTING:

# Random Forest Classifier for the numeric encoded data:
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Display performance results
print("Accuracy for Numeric Encoded Model:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=definition_list_MCI))

# Call the confusion matrix visualization
plot_confusion_matrix(y_test, y_pred, definition_list_MCI)

#--------------------------------------------------#

# One Hot Encoding categorical variables for modeling:
categorical_columns = ['PREMISES_TYPE', 'OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOY', 'OCC_DOW', 'OCC_HOUR', 'DIVISION', 'HOOD_158']

# Apply OneHotEncoder to categorical columns
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(sparse_output=False), categorical_columns)],
    remainder='passthrough'  # Keep numerical columns as is
)

# Fit and transform the OneHotEncoder to X_train and X_test
X_train_OH = preprocessor.fit_transform(X_train)
X_test_OH = preprocessor.transform(X_test)

# Random Forest Classifier for the One Hot Encoded data:
classifier_OH = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
classifier_OH.fit(X_train_OH, y_train)
y_pred_OH = classifier_OH.predict(X_test_OH)

# Display performance results for the One Hot Encoded model
print("Accuracy for One Hot Encoded Model:", accuracy_score(y_test, y_pred_OH))
print(classification_report(y_test, y_pred_OH, target_names=definition_list_MCI))

# Call the confusion matrix visualization for One Hot Encoded model
plot_confusion_matrix(y_test, y_pred_OH, definition_list_MCI)

#--------------------------------------------------#

# Balanced Class Weight doesn't make a big difference for results:
classifier_balanced = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42, class_weight='balanced')
classifier_balanced.fit(X_train, y_train)
y_pred_balanced = classifier_balanced.predict(X_test)

# Display performance for the balanced class weight model
print("Accuracy for Balanced Class Weight Model:", accuracy_score(y_test, y_pred_balanced))
print(classification_report(y_test, y_pred_balanced, target_names=definition_list_MCI))

# Call the confusion matrix visualization for balanced model
plot_confusion_matrix(y_test, y_pred_balanced, definition_list_MCI)

#--------------------------------------------------#

# Gradient Boosting Model:
grad_class = GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, random_state=42)
grad_class.fit(X_train_OH, y_train)
y_pred_grad = grad_class.predict(X_test_OH)

# Display performance results for Gradient Boosting model
print("Accuracy for Gradient Boosting Model:", accuracy_score(y_test, y_pred_grad))
print(classification_report(y_test, y_pred_grad, target_names=definition_list_MCI))

# Call the confusion matrix visualization for Gradient Boosting model
plot_confusion_matrix(y_test, y_pred_grad, definition_list_MCI)
