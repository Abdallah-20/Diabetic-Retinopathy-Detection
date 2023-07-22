from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, \
   QMessageBox
from PyQt5.QtGui import  QIcon
from PyQt5.QtCore import Qt
from pymongo import MongoClient
from NewDetection import Example
import  datetime , random
visitdate=datetime.datetime.now()
class RegistrationForm(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Patient Data')
        self.setFixedSize(700, 500)
        self.setStyleSheet("background-color:white")
        self.setWindowIcon(QIcon('C:/Users/DELL/Downloads/real-brown-eye-png-10.png'))

        self.title_label = QLabel('Enter Patient Data')
        self.title_label.setAlignment(Qt.AlignLeft)
        self.title_label.setStyleSheet('font-weight: bold; color: #4E668F; font-size: 20px; margin-top:30px;'
                                       ' margin-bottom:45px; margin-left:25px;')

        self.name_label = QLabel('Name:')
        self.name_input = QLineEdit()
        self.name_input.setStyleSheet('margin-bottom:25px; margin-left:25px;font-size:20px;'
                                      'max-width:500px;border-radius:5px;''background-color:#D9D9D9;')
        self.name_label.setStyleSheet('color: #30476F; font-weight:bold;font-size:17px; margin-left:25px;')

        self.age_label = QLabel('Age:')
        self.age_input = QLineEdit()
        self.age_input.setStyleSheet('margin-bottom:25px; margin-left:25px;font-size:20px;'
                                      'max-width:500px;border-radius:5px;''background-color:#D9D9D9;')
        self.age_label.setStyleSheet('color: #30476F; font-weight:bold;font-size:17px; margin-left:25px;')

        self.phone_label = QLabel('Phone:')
        self.phone_input = QLineEdit()
        self.phone_input.setStyleSheet('margin-bottom:35px;margin-left:25px;font-size:20px;'
                                      'max-width:500px;border-radius:5px;''background-color:#D9D9D9;')
        self.phone_label.setStyleSheet('color: #30476F; font-weight:bold;font-size:17px; margin-left:25px;')

        self.submit_button = QPushButton('Submit')
        self.submit_button.setFixedWidth(200)
        self.submit_button.setStyleSheet('background-color: #3A7FF6; color: white; font-weight: bold; font-size: 14px;'
                                    ' border-radius:15px; padding:7px; margin-top:15px;max-width:187px;')
        self.submit_button.clicked.connect(self.submit_form)


        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)
        layout.addWidget(self.age_label)
        layout.addWidget(self.age_input)
        layout.addWidget(self.phone_label)
        layout.addWidget(self.phone_input)
        layout.addWidget(self.submit_button)
        layout.setAlignment(self.submit_button, Qt.AlignCenter)
        layout.addStretch()
        self.setLayout(layout)

        self.client = MongoClient('mongodb://localhost:27017')
        self.db = self.client['DiabeticRetinopathy']
        self.collection = self.db['Patient']
        self.collection.distinct('PatientCode')
    def submit_form(self):
        name = self.name_input.text().strip()
        age = self.age_input.text().strip()
        phone = self.phone_input.text().strip()
        visit = str(visitdate.year)+"-"+str(visitdate.strftime("%B"))+"-"+str(visitdate.day)
        code = str(random.randint(1000000, 9999999))
        if not name or not age or not phone:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setInformativeText("Please fill all data fields")
            msg.setWindowTitle("ERROR")
            msg.setStyleSheet('width:350px;')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        elif name.isalnum() == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setInformativeText("Name should not contain special characters")
            msg.setWindowTitle("ERROR")
            msg.setStyleSheet('width:350px;')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        elif name.isnumeric() == True:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setInformativeText("Name should be strings not numbers")
            msg.setWindowTitle("ERROR")
            msg.setStyleSheet('width:350px;')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        elif age.isnumeric() ==False or phone.isnumeric() == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setInformativeText("Age and Phone must be numbers")
            msg.setWindowTitle("ERROR")
            msg.setStyleSheet('width:350px;')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        elif int(age) <= 0 or int(age) >= 110:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setInformativeText("Only accepts age from 1 to 110")
            msg.setWindowTitle("ERROR")
            msg.setStyleSheet('width:350px;')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        elif len(phone) != 11:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setInformativeText("Phone number must contain 11 numbers")
            msg.setWindowTitle("ERROR")
            msg.setStyleSheet('width:350px;')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        else:
            data = {'PatientCode':code,'Name': name, 'Age': age, 'Phone': phone,'Visit': visit,"Class":"","Level":""}
            self.collection.insert_one(data)
            self.newwin = Example()
            self.newwin.show()
            self.newwin.patientid_edit.setText(code)
            self.newwin.patientid_edit.setReadOnly(True)
            self.close()


if __name__ == '__main__':
    app = QApplication([])
    form = RegistrationForm()
    form.show()
    app.exec_()
