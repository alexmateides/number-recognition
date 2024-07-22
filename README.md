# number-recognition
![img1](https://github.com/alexmateides/number-recognition/blob/main/assets/mnist2.png)

## About
I originally developed this application back in autumn 2022 in my last year of high school as a personal project. 
It was my first bigger Machine Learning project and I learned a lot about ML, CNNs and image processing while developing this app. I originally used TensorFlow2 for ML and TKinter for GUI. Now I have modernised it to use PyTorch and PyQT6

## Installation
You will need PyTorch and PyQT6

**1. Install dependencies**
```sh
pip install -U torch PyQt6
```

**2. Clone the repo**
```sh
git clone https://github.com/alexmateides/number-recognition
```

**3. Change WD**
```sh
cd number-recognition
```

**4. Start the app**
```sh
python -m app
```

## Usage

![img2-gui](https://github.com/alexmateides/number-recognition/blob/main/assets/UI.png)

## Model
`./train/` contains the original training script in TensorFlow2 and the new training script from [Google](https://colab.research.google.com/github/rpi-techfundamentals/fall2018-materials/blob/master/10-deep-learning/04-pytorch-mnist.ipynb) that can be ran on free Colab T4 instance. 
I modified the google script by changing the optimizer to AdamW (gives much better results) and lowered number of epochs to 5 (to prevent overfitting)

## Files

```md
number-recognition
│
├── assets                  # Example images
│
├── src                     # Source code and PyTorch model
│
├── train                   # Contains training scripts
│
├── LICENSE                 # Open source license
│
└── README.md               # This file
```


## License
Distributed under the MIT License. See LICENSE for more information.

## Contact
**Alexander Mateides** - alex.mateides@gmail.com - [LinkedIn](https://www.linkedin.com/in/alexander-mateides-138136285/)

