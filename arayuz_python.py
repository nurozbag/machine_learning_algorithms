import sys
from arayuz import Ui_Form
from PyQt5.QtWidgets import *
import pandas as pd
from knn_arayuz_python import KNN_ArayuzPage
from decision_tree_python import Decision_Tree
from svm_python import SVM
from ann_python import ANN
import joblib
import numpy as np
import pickle

class ArayuzPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.knn_sayfa = KNN_ArayuzPage()
        self.svm_sayfa = SVM()
        self.dt_sayfa = Decision_Tree()
        self.ann_sayfa = ANN()
        self.data_set = []
        self.ui.BtnDYukle.clicked.connect(self.data_yukle)
        self.ui.BtnKNN.clicked.connect(self.knn_git)
        self.ui.BtnDT.clicked.connect(self.dt_git)
        self.ui.BtnSVM.clicked.connect(self.svm_git)
        self.ui.BtnANN.clicked.connect(self.ann_git)
        self.ui.BtnReal.clicked.connect(self.total)
        self.ui.tableWidget.cellClicked.connect(self.on_cell_clicked)
    def data_yukle(self):
        try:
            data_set = pd.read_csv("C:/Users/Pc/Desktop/ML_project/Epileptic Seizure Recognition.csv")
           

            if not data_set.empty:
                self.populateTable(data_set)
            else:
                print("CSV file is empty.")
        except pd.errors.EmptyDataError:
            print("CSV file is empty.")
        except pd.errors.ParserError:
            print("Error parsing CSV file.")           
    def populateTable(self, df):
        self.ui.tableWidget.clear()
        # Set row and column count
        self.ui.tableWidget.setRowCount(df.shape[0])
        self.ui.tableWidget.setColumnCount(df.shape[1])
        self.ui.tableWidget.setHorizontalHeaderLabels(df.columns)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[i, j]))
                self.ui.tableWidget.setItem(i, j, item)
                
    def on_cell_clicked(self, row, col):
        selected_row = self.ui.tableWidget.currentRow()
        self.data_set = []
        for inner_col in range(self.ui.tableWidget.columnCount()):
            item = self.ui.tableWidget.item(selected_row, inner_col)
            self.data_set.append(item.text())
                
    
    def total(self):
        self.ui.label.setText('-')
        with open('C:/Users/Pc/Desktop/ML_project/withclfdt.py','wb') as model_file:
            model_dt=pickle.load(model_file)
        tahmin_dt=model_dt.predict(dataset)    
        result=np.argmax(np.sum[tahmin_dt],axis=0)
        self.ui.label.setText(f"siniflandirici:{result}")
        
           
    
    def knn_git(self):
        self.knn_sayfa.show()
    def dt_git(self):
        self.dt_sayfa.show()
    def svm_git(self):
        self.svm_sayfa.show()
    def ann_git(self):
        self.ann_sayfa.show()
    
if __name__ == '__main__':
   app = QApplication(sys.argv)
   pencere = ArayuzPage()
   pencere.show()
   sys.exit(app.exec_())
