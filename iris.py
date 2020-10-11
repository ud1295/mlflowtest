from __future__ import print_function #dev2

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import metrics
import mlflow
import mlflow.sklearn

if __name__ == '__main__':
    # mlflow.create_experiment("mlflowproject_demo1")
    # mlflow.set_experiment("mlflowproject_demo1")
    X, y = load_iris(return_X_y=True)
    clf = svm.SVC(kernel='linear', C=10)
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
    mlflow.log_params({'kernel': 'linear', 'C':10})
    mlflow.log_metrics({"score": scores.mean(), 'score2':scores[0]})
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # mlflow.sklearn.log_model(clf, "iris classification model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
    mlflow.sklearn.log_model(clf, "SVM_CLF_model")

    # mlflow.sklearn.save_model(clf, 'iris_SVM_clf')
    # saved this model for mlflow 
    # adding random line for testing

    # mdl = mlflow.pyfunc.load_model(model_path)
