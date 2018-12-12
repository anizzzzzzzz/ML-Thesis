## Getting Started

These instructions will get you a copy of the project up and running on your local machine for training, creating model and classifying the content. 

### Prerequisities
Need python3.6 version.

### Creating Virtual Environment
To create a virtual environment, go to your project’s directory and run virtualenv.

#### On macOS and Linux:
``` 
python3 -m virtualenv venv
``` 
#### On windows:
``` 
py -m virtualenv venv
```
Note : The second argument is the location to create the virtualenv. Generally, you can just create this in your project and call it venv.
[ virtualenv will create a virtual Python installation in the venv folder.]

### Activating Virtual Environment
Before you can start installing or using packages in your virtualenv, you’ll need to activate it.

#### On macOS and Linux:
```
source venv/bin/activate
```

#### On windows:
```
.\venv\Scripts\activate
```

#### Confirming virtualenv by checking location

##### On macOS and Linux:
```
which python
```
Output : .../venv/bin/python

##### On windows:
```
where python
```
Output : .../venv/bin/python.exe



### Installing packages with pip
```
pip install -r requirements.txt
```
