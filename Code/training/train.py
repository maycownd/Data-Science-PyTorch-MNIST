import joblib
import logging
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition.pca import PCA
import sys
import os
import time

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from preprocessing.preprocess import import_xy


MODEL_PATH = "../../Data/models/latest_model.pkl"


def train_model():
    X_train, y_train, X_test, y_test = import_xy()
    X_train /= 255.0
    X_test /= 255.0

    # model = SVC(kernel='linear')
    model = make_pipeline(PCA(500),
                          LGBMClassifier(random_state=33))
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n", classification_report(y_test, y_pred))
    return model, accuracy, (end_time-start_time)


def save_model(model, metric, model_path=MODEL_PATH):

    print(round(metric, 2))
    joblib.dump(model, model_path)


if __name__ == "__main__":
    model, accuracy, time_train = train_model()
    print(time_train)
    save_model(model, accuracy)
    print("Success!")
