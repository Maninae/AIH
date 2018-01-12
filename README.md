# AIH
Stanford AI + Healthcare research sample. Winter 17-18.
Built using Keras, in Python 3.

Training and inference scripts are located at ```training.py``` and ```inference.py```.

# Files under ```utils```
```image.py``` contains code to visualize depth maps, plot loss and accuracy for models.
```data.py``` has methods for loading data for inference and training, prompts for user input during various scripts.
```debug.py``` contains one debug function, in case it is needed during development.
```test.py``` is for quick testing of methods as-needed.

# The ```model``` directory
```model.py``` contains the CNN model. It also has a custom activation (the Swish unit parametrized with beta=1) and hacked precision and recall metrics.
```completed``` contains subdirs named according to a sensor, or 'all'. Each subdir contains a model trained on the data specifically for that sensor; the directory ```all``` contains a general purpose model trained on the training partition of all 113K+ samples (used for sensors that do not have a model specifically designated here under a subdir of ```model```).

# The ```dataset``` directory
The sensor data is here. ```nopos``` has the sensors with no positive (1) samples. ```imbalanced``` has the ones where the proportion of positive to negative examples is < 0.05.
```partition_script.py``` has the code to make a 70/30 training-dev split for each sensor.

# The ```assets``` directory
This folder contains the pdf for the research sample. It also has figures for loss/accuracy during training for the sensors' data.
