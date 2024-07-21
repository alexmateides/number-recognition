import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton, QLineEdit
from PyQt6.QtGui import QPainter, QPen, QImage
from PyQt6.QtCore import Qt, QEvent, QPoint

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Define your model class here
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # Dropout
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Convolutional Layer/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Convolutional Layer/Dropout/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        # Fully Connected Layer/Activation
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # Fully Connected Layer/Activation
        x = self.fc2(x)
        # Softmax gets probabilities.
        return F.log_softmax(x, dim=1)

class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super(DrawingWidget, self).__init__(parent)
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        self.drawing = False
        self.lastPoint = None
        self.setFixedSize(280, 280)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.lastPoint = event.position()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            pen = QPen(Qt.GlobalColor.black, 30, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.lastPoint, event.position())
            self.lastPoint = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def resizeEvent(self, event):
        newImage = QImage(event.size(), QImage.Format.Format_RGB32)
        newImage.fill(Qt.GlobalColor.white)
        painter = QPainter(newImage)
        painter.drawImage(QPoint(), self.image)
        self.image = newImage

    def clearImage(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def getImage(self):
        return self.image

class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.setFixedSize(280, 280)

    def setImage(self, image):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        self.canvas.draw()

    def clearImage(self):
        self.figure.clear()
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('MNIST Handwriting')

        self.drawingWidget = DrawingWidget()
        self.imageWidget = ImageWidget()
        self.clearButton = QPushButton('Clear')
        self.clearButton.clicked.connect(self.clearDrawing)

        self.predictButton = QPushButton('Predict')
        self.predictButton.clicked.connect(self.predictDigit)

        self.predictedNumber = QLineEdit()
        font = self.font()
        font.setBold(True)
        font.setPointSizeF(30)
        self.predictedNumber.setFont(font)
        self.predictedNumber.setReadOnly(True)
        self.predictedNumber.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)

        self.resultLabel = QLabel('Prediction: ')
        self.resultLabel.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        self.resultLabel.setFixedHeight(200)  # Adjusted for more space

        mainLayout = QVBoxLayout()
        drawingLayout = QHBoxLayout()
        drawingLayout.addWidget(self.drawingWidget)
        drawingLayout.addWidget(self.imageWidget)
        mainLayout.addLayout(drawingLayout)
        mainLayout.addWidget(self.clearButton)
        mainLayout.addWidget(self.predictButton)
        mainLayout.addWidget(self.predictedNumber)
        mainLayout.addWidget(self.resultLabel)

        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        self.drawingWidget.installEventFilter(self)

        self.model = self.loadModel()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def loadModel(self):
        # Load the entire model object
        model = torch.load('mnist_model.pt', map_location=torch.device('cpu'))
        model.eval()
        return model

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.MouseButtonRelease and source is self.drawingWidget:
            self.updatePreview()
        return super(MainWindow, self).eventFilter(source, event)

    def updatePreview(self):
        grayscale = self.getGrayscaleImage()
        self.imageWidget.setImage(grayscale)

    def clearDrawing(self):
        self.drawingWidget.clearImage()
        self.imageWidget.clearImage()
        self.resultLabel.setText('Prediction: ')
        self.predictedNumber.clear()

    def getGrayscaleImage(self):
        drawnImage = self.drawingWidget.getImage()
        smallImage = drawnImage.scaled(28, 28, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        buffer = smallImage.bits().asstring(smallImage.width() * smallImage.height() * smallImage.depth() // 8)
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((28, 28, 4))
        rgb_arr = arr[:, :, :3]

        # Convert RGB to Grayscale
        r, g, b = rgb_arr[:, :, 0], rgb_arr[:, :, 1], rgb_arr[:, :, 2]
        grayscale = 0.299 * r + 0.587 * g + 0.114 * b

        return grayscale

    def predictDigit(self):
        grayscale = self.getGrayscaleImage()

        # Convert to PIL image and apply the same transformations as during training
        pil_image = transforms.ToPILImage()(grayscale).convert("L")
        tensor = self.transform(pil_image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = self.model(tensor)
            print(output)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            print(pred)
            probabilities = torch.exp(output).squeeze().numpy() * 100

        prediction_text = 'Prediction:\n'
        for i, prob in enumerate(probabilities):
            prediction_text += f'{i} - {prob:.2f}%\n'

        self.resultLabel.setText(prediction_text)
        self.predictedNumber.setText(str(pred.item()))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())
