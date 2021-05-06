#Linear Discriminant Analysis With scikit-learn
from functions import *     #imports our functions

# Get Data
X = get_data('combined_csv.csv')
y = get_data('combined_csv.csv')

# Format Data
del X['SleepLVL'] 
del y['HR'] 

#Create model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train.values.ravel())
X_test = lda.transform(X_test)

#Predict values
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train.values.ravel())
y_pred = classifier.predict(X_test)

#Check accuracy
cm = confusion_matrix(y_test, y_pred)

# Convert pandas df to numpy array
y_testnp = np.asarray(y_test)
y_prednp = np.asarray(y_pred)

# Display Accuracy
display(Accuracy_of_actual(y_testnp, y_prednp))
display(Accuracy_of_awake(y_testnp, y_prednp))