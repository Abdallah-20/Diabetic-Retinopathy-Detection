import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QFrame, QVBoxLayout, QLabel, QPushButton, QLineEdit, \
    QMessageBox, QFileDialog
from pymongo import MongoClient
from keras import models
import tensorflow as tf
import numpy as np

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

        self.welcome_label = QLabel("Upload Image", frame1)
        self.welcome_label.setStyleSheet(
            "color: white; font-size: 20px; font-family:Inter;font-weight: bold; padding-bottom:10px; margin-top:30px; border-bottom: 2px solid white;")
        self.welcome_label.setAlignment(Qt.AlignLeft)

        self.imageLabel = QLabel(self)
        self.imageLabel.setFixedSize(350, 250)
        self.imageLabel.setStyleSheet('border: 2px solid #D9D9D9; background-color:#D9D9D9;')

        self.uploadButton = QPushButton('Upload Image', self)
        self.uploadButton.setStyleSheet('background-color: white; color: #4E668F; font-weight: bold; font-size: 14px;'
                                         ' border-radius:15px; padding:7px; margin-top:25px; max-width:187px; margin-left:65px;')
        self.uploadButton.clicked.connect(self.uploadImage)

        frame1_layout = QVBoxLayout(frame1)
        frame1_layout.addWidget(self.welcome_label)
        frame1_layout.addWidget(self.imageLabel)
        frame1_layout.addWidget(self.uploadButton)
        frame1_layout.addStretch()

        self.title = QLabel("Enter Your Information", frame2)
        self.title.setAlignment(Qt.AlignHCenter)
        self.title.setStyleSheet("margin-top:15px; font-size:20px; color:#4E668F")
        self.patientid_label = QLabel("Patient ID:", frame2)
        self.patientid_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px; color:#30476F;")
        self.patientid_edit = QLineEdit(frame2)
        self.patientid_edit.setStyleSheet('margin-bottom:25px;  font-size:20px; '
                                          'boder: 2px solid gray;border-radius:5px; background-color:#D9D9D9;')
        self.diseaseclassification_label = QLabel("Disease Classification:", frame2)
        self.diseaseclassification_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px; color:#30476F")
        self.diseaseclassification_edit = QLineEdit(frame2)
        self.diseaseclassification_edit.setStyleSheet(
            ('margin-bottom:20px;  font-size:20px; boder: 2px solid gray;border-radius:5px; background-color:#D9D9D9;'))
        self.diseaseclassification_edit.setReadOnly(True)
        self.diseaselevel_label = QLabel("Disease level:", frame2)
        self.diseaselevel_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px; color:#30476F")
        self.diseaselevel_edit = QLineEdit(frame2)
        self.diseaselevel_edit.setStyleSheet(
            ('margin-bottom:20px;  font-size:20px; boder: 2px solid gray;border-radius:5px; background-color:#D9D9D9;'))
        self.diseaselevel_edit.setReadOnly(True)
        self.submit_button = QPushButton("Submit", frame2)
        self.submit_button.setStyleSheet('background-color: #3A7FF6; color: white; font-weight: bold; font-size: 14px;'
                                         ' border-radius:15px; padding:7px; margin-top:15px; max-width:187px; margin-left:65px;')
        self.submit_button.clicked.connect(self.submit_form)

        self.back_button = QPushButton("Back", frame2)
        self.back_button.setStyleSheet('background-color: #3A7FF6; color: white; font-weight: bold; font-size: 14px;'
                                         ' border-radius:15px; padding:7px; margin-top:5px; max-width:187px; margin-left:65px;')
        self.back_button.clicked.connect(self.back_page)

        frame2_layout = QVBoxLayout(frame2)
        frame2_layout.addWidget(self.title)
        frame2_layout.addWidget(self.patientid_label)
        frame2_layout.addWidget(self.patientid_edit)
        frame2_layout.addWidget(self.diseaseclassification_label)
        frame2_layout.addWidget(self.diseaseclassification_edit)
        frame2_layout.addWidget(self.diseaselevel_label)
        frame2_layout.addWidget(self.diseaselevel_edit)
        frame2_layout.addWidget(self.submit_button)
        frame2_layout.addWidget(self.back_button)
        frame2_layout.addStretch()

        layout.addWidget(frame1)
        layout.addWidget(frame2)

        self.client = MongoClient('mongodb://localhost:27017')
        self.db = self.client['DiabeticRetinopathy']
        self.collection = self.db['Patient']
        self.setFixedSize(750, 500)

        self.setWindowTitle('Patient Data')
        self.setWindowIcon(QIcon('C:/Users/DELL/Downloads/real-brown-eye-png-10.png'))
        self.show()

    def update_document(self,document_id, new_values):
        self.collection.find_one_and_update({'PatientCode':document_id},{'$set':new_values})
    def submit_form(self):
        document_id = self.patientid_edit.text().strip()
        level = self.diseaselevel_edit.text().strip()
        diseaseclass = self.diseaseclassification_edit.text().strip()
        if not document_id or not level or not diseaseclass:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setInformativeText("All Fields must be filled")
            msg.setWindowTitle("ERROR")
            msg.setStyleSheet('width:350px;')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        else:
            new_values = {'Class': self.diseaseclassification_edit.text().strip(), 'Level': self.diseaselevel_edit.text().strip()}
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setInformativeText("Patient Data Updated")
            msg.setWindowTitle("Success")
            msg.setStyleSheet('width:350px;')
            msg.setStandardButtons(QMessageBox.Ok)
            self.update_document(document_id, new_values)


    def uploadImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.bmp)')
        if fileName:
            pixmap = QPixmap(fileName)
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.width(), self.imageLabel.height()))
            self.predict(fileName)
        else:
            print("No Such Directory")

    def predict(self,filename):
        model_eval = models.load_model("D:\Materials\Graduation Project\Experimental Results\Resnet152V2\ResNet152V2.h5")
        model_eval.compile(loss='categorical_crossentropy',
                           optimizer='Adam',
                           metrics=['accuracy'])
        image = tf.keras.utils.load_img(filename, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model_eval.predict(images)
        res = classes.argmax()
        i = " "
        if (res == 0):
            i = "Normal"
        elif (res == 1):
            i = "Mild"
        elif (res == 2):
            i = "Moderate"
        elif (res == 3):
            i = "Severe"
        else:
            i = "Proliferative"
        self.diseaselevel_edit.setText(str(res))
        self.diseaseclassification_edit.setText(i)
        self.diseaselevel_edit.setReadOnly(True)
        self.diseaseclassification_edit.setReadOnly(True)
        return i,res
    def back_page(self):
        from NewPatientData import RegistrationForm
        self.newwin = RegistrationForm()
        self.newwin.show()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
