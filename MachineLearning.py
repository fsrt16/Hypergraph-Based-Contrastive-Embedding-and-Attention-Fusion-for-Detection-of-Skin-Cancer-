import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Classifiers
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    BaggingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class MLModelTrainer:
    def __init__(self, xop, y, classes, base_dir="Results"):
        self.xop = xop
        self.y = y
        self.classes = classes
        self.processing_times = {}
        self.base_dir = base_dir
        self.ml_dir = os.path.join(base_dir, 'ML')
        X_train, X_test, y_train, y_test = train_test_split(
            self.xop, self.y, test_size=0.2, random_state=42, stratify=self.y )
        # Initialize models
        self.models = self._initialize_models()
        self._prepare_dirs()
    @classmethod
    def from_splits(cls, X_train, X_test, y_train, y_test, classes, base_dir="Results"):
        """Alternative initializer when you already have train/test splits."""
        obj = cls.__new__(cls)  # create an instance without calling __init__

        obj.X_train = X_train
        obj.X_test = X_test
        obj.y_train = y_train
        obj.y_test = y_test
        obj.classes = classes
        obj.processing_times = {}
        obj.base_dir = base_dir
        obj.ml_dir = os.path.join(base_dir, 'ML')

        obj.models = obj._initialize_models()
        obj._prepare_dirs()
        return obj
    
    def _initialize_models(self):
        return {
            "LogisticRegression": LogisticRegression(max_iter=200),
            "RandomForest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "GradientBoosting": GradientBoostingClassifier(),
            "MLPClassifier": MLPClassifier(max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(),
            "NaiveBayes": GaussianNB(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": LGBMClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0),
            "KNN": KNeighborsClassifier(),
            "ExtraTrees": ExtraTreesClassifier(),
            "LDA": LinearDiscriminantAnalysis(),
            "Bagging": BaggingClassifier(),
            # Extended versions
            "ShallowMLP": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000),
            "DeepMLP": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000),
            "WideMLP": MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=1000),
            "DropoutMLP": MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.001, max_iter=1000),
            "TanhMLP": MLPClassifier(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=1000),
            "LogisticMLP": MLPClassifier(hidden_layer_sizes=(100, 50), activation='logistic', max_iter=1000),
        }
    def _prepare_dirs(self):
        os.makedirs(self.ml_dir, exist_ok=True)

    def train_all_models(self):
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test 
        overall_metrics = []

        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")

            start_time = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = self._safe_predict_proba(model, X_test)
            end_time = time.time()

            processing_time = end_time - start_time
            self.processing_times[model_name] = processing_time

            self._save_classification_report(model_name, y_test, y_pred)
            self.plot_classification_metrics(y_test, y_pred, model_name, save=True)
            self.plot_confusion_matrix(y_test, y_pred, model_name, save=True)

            if y_prob is not None:
                self.plot_roc_curve(y_test, y_prob, model_name, save=True)

            metrics = self._compute_overall_metrics(y_test, y_pred, y_prob, model_name)
            metrics['Processing Time (s)'] = round(processing_time, 2)
            overall_metrics.append(metrics)

        self.results = pd.DataFrame(overall_metrics)
        self.results.to_csv(os.path.join(self.ml_dir, "overall_metrics.csv"), index=False)
        print("\n=== Overall Metrics ===")
        print(self.results)

    def _safe_predict_proba(self, model, X_test):
        """Handles models that may not support predict_proba."""
        try:
            return model.predict_proba(X_test)
        except AttributeError:
            print(f"Model {model.__class__.__name__} does not support predict_proba(). Skipping ROC curve.")
            return None

    def _save_classification_report(self, model_name, y_test, y_pred):
        report = classification_report(y_test, y_pred, target_names=self.classes, output_dict=True)
        print(f"Classification report for {model_name}:")
        print(classification_report(y_test, y_pred, target_names=self.classes))
        print("\n")
        df_report = pd.DataFrame(report).transpose()
        path = os.path.join(self.ml_dir, f"{model_name}_classification_report.csv")
        df_report.to_csv(path)
        print(f"Saved classification report: {path}")

    def _compute_overall_metrics(self, y_true, y_pred, y_prob, model_name):
        acc = np.mean(y_true == y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(
                    label_binarize(y_true, classes=range(len(self.classes))), y_prob,
                    average='macro', multi_class='ovr'
                )
            except ValueError:
                roc_auc = np.nan
        else:
            roc_auc = np.nan

        return {
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "ROC AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A"
        }

    def plot_classification_metrics(self, y_true, y_pred, model_name, save=False):
        report = classification_report(y_true, y_pred, output_dict=True, target_names=self.classes)
        classes = self.classes
        precision = [report[c]['precision'] for c in classes]
        recall = [report[c]['recall'] for c in classes]
        f1 = [report[c]['f1-score'] for c in classes]
        support = [report[c]['support'] for c in classes]

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(classes, support, color='lightgrey', label='Support')
        ax1.set_ylabel('Support', color='black')
        ax2 = ax1.twinx()
        ax2.plot(classes, precision, 'b-o', label='Precision')
        ax2.plot(classes, recall, 'r-o', label='Recall')
        ax2.plot(classes, f1, 'g-o', label='F1-Score')
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel('Score', color='black')
        # x label should be 90 degrees 
        ax1.set_xlabel('Classes', color='black')
        ax1.tick_params(axis='x', rotation=90)
        plt.title(f'Classification Metrics for {model_name}')
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

        if save:
            path = os.path.join(self.ml_dir, f"{model_name}_metrics_plot.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics plot: {path}")
        plt.show()
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, model_name, save=False):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'Confusion Matrix for {model_name}')
        
        if save:
            path = os.path.join(self.ml_dir, f"{model_name}_confusion_matrix.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix: {path}")
        plt.show()
        plt.close()

    def plot_roc_curve(self, y_test, y_prob, model_name, save=False):
        n_classes = len(self.classes)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        plt.figure(figsize=(8, 6))

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f'Class {self.classes[i]} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc="lower right")

        if save:
            path = os.path.join(self.ml_dir, f"{model_name}_roc_curve.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved ROC curve: {path}")
        plt.show()
        plt.close()
