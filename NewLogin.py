import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QFrame, QVBoxLayout, QLabel, QPushButton, QLineEdit, \
    QMessageBox
from pymongo import MongoClient
from NewPatientData import RegistrationForm

class Example(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout(self)
        layout.setSpacing(0)
        frame1 = QFrame(self)
        frame1.setFixedWidth(375)
        frame2 = QFrame(self)
        frame1.setStyleSheet("background-color: #3A7FF6")
        frame2.setStyleSheet("background-color: white")

        self.welcome_label = QLabel("Welcome To Our Program", frame1)
        self.welcome_label.setStyleSheet("color: white; font-size: 20px; font-family:Inter;font-weight: bold; padding-bottom:10px; margin-top:70px; border-bottom: 2px solid white;")
        self.welcome_label.setAlignment(Qt.AlignLeft)

        self.additional_label = QLabel("Diabetes affect your eyes in many ways.\n"
                                  "Some people go on to develop an eye com-\n"
                                  "plication called diabetic retinopathy.\n"
                                  "if you don't get this eye problem treated in\n"
                                  "time, it can lead to sight loss.", frame1)
        self.additional_label.setStyleSheet("color: white; font-size: 18px; font-family:Inter; margin-top:15px;  qproperty-indent:0")
        self.additional_label.setAlignment(Qt.AlignLeft)

        frame1_layout = QVBoxLayout(frame1)
        frame1_layout.addWidget(self.welcome_label)
        frame1_layout.addWidget(self.additional_label)
        frame1_layout.addStretch()

        self.title = QLabel("Enter Your Information",frame2)
        self.title.setAlignment(Qt.AlignHCenter)
        self.title.setStyleSheet("margin-top:70px; font-size:20px; color:#4E668F")
        self.doctorid_label = QLabel("Doctor ID:", frame2)
        self.doctorid_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px; color:#30476F;")
        self.doctorid_edit = QLineEdit(frame2)
        self.doctorid_edit.setStyleSheet('margin-bottom:25px;  font-size:20px; boder: 2px solid gray;border-radius:5px; background-color:#D9D9D9;')
        self.password_label = QLabel("Password:", frame2)
        self.password_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px; color:#30476F")
        self.password_edit = QLineEdit(frame2)
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setStyleSheet(('margin-bottom:20px;  font-size:20px; boder: 2px solid gray;border-radius:5px; background-color:#D9D9D9;'))

        self.submit_button = QPushButton("Login", frame2)
        self.submit_button.setStyleSheet('background-color: #3A7FF6; color: white; font-weight: bold; font-size: 14px;'
                                    ' border-radius:15px; padding:7px; margin-top:15px; max-width:187px; margin-left:65px;')
        self.submit_button.clicked.connect(self.submit_form)

        frame2_layout = QVBoxLayout(frame2)
        frame2_layout.addWidget(self.title)
        frame2_layout.addWidget(self.doctorid_label)
        frame2_layout.addWidget(self.doctorid_edit)
        frame2_layout.addWidget(self.password_label)
        frame2_layout.addWidget(self.password_edit)
        frame2_layout.addWidget(self.submit_button)
        frame2_layout.addStretch()

        layout.addWidget(frame1)
        layout.addWidget(frame2)

        self.client = MongoClient('mongodb://localhost:27017')
        self.db = self.client['DiabeticRetinopathy']
        self.collection = self.db['Doctor']
        self.setFixedSize(750, 500)
        self.setWindowTitle('Login')
        self.setWindowIcon(QIcon('C:/Users/DELL/Downloads/real-brown-eye-png-10.png'))
        self.show()

    def submit_form(self):
        doctorid = self.doctorid_edit.text().strip()
        password = self.password_edit.text().strip()
        if not doctorid or not password:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setInformativeText("Please fill all data fields")
            msg.setWindowTitle("ERROR")
            msg.setStyleSheet('width:350px;')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        else:
            cursor = self.collection.find({})
            for document in cursor:
                if document['DoctorId'] == doctorid and document['Password'] == password:
                    self.newwin = RegistrationForm()
                    self.newwin.show()
                    self.close()
                else:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setInformativeText("Wrong ID or Password")
                    msg.setWindowTitle("ERROR")
                    msg.setStyleSheet('width:350px;')
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec()
if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())