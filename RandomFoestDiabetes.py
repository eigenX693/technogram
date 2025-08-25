import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
#encoding target variable
le = LabelEncoder()
#Bringing in the data set and removing the unsuable fields like id etc
dataset = pd.read_csv(r'C:\Users\eigenX\Desktop\BACKUP\Data_Analytics\DatasetofDiabetes.csv')
x = dataset.drop(['ID','No_Pation','CLASS'],axis=1)
y =le.fit_transform(dataset['CLASS'].values.ravel())  #1D array
#y = dataset.iloc[:,9:10].values
#ID= dataset.iloc[:,0:1].values
x['Gender'] = le.fit_transform(x['Gender']) #encoding gender f=0 m=1
#splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
#ID= dataset.iloc[:len(y_test),0:1]
#print(dataset.head())
#now fo the best part...Initiating the model
model = RandomForestClassifier(n_estimators=200,max_depth = 10,class_weight= 'balanced',random_state= 42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("-------------------------------------------------------------------")
print("-------c-L-A-S-S-I-F-I-C-A-T-I-O-N----R-E-P-O-R-T------------------")
#print(classification_report(y_test, y_pred, target_names=['N','P','Y']))
print("Class.Info.Report: \n", classification_report(y_test, y_pred, target_names=['N','P','Y','Unknown']))
#building a confusion matrix
conf_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix, annot=True ,fmt='d',cmap='Blues',xticklabels = ['N','P','Y'], yticklabels=['N','P','Y'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()
print("ACCURACY: ",accuracy_score(y_test,y_pred))
print("Weighted F1 SCORE: ",f1_score(y_test,y_pred, average = 'weighted'))