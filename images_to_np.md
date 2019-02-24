
In this draft, we are going to convert images to one numpy matrix. The requirements are that the images should be saved in a folder. We will read in all the images to separate numpy matices and flatten these matrices to make a large row vector and stack these rows so that in the final numpy matrix each row is an image and columns are corresponding to pixels. 


import the necessary libraries first:


```python
import numpy as np
import pandas as pd
import csv

import cv2

from sklearn.model_selection import train_test_split
```

Define a function which gets a image file name and load the image to a numeric matrix and store it in a numpy array:


```python
# im = cv2.imread("./example.TIF")

from PIL import Image
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray(img, dtype="int32" )
    return data
```

Using pathlib library, we can read all of files in a directory. Since we have many images in a directory, we need to read in the images in a loop. This can be done through this library. Noted that my images end up with .TIF extension. You should change it according to the sufices of your images:


```python
from pathlib import Path

path = Path.cwd()

path = Path("/home/mr/SSC_case_study/trainexamples/")

img_pths = [pth for pth in path.iterdir() if pth.suffix == '.TIF']
```

Since the dimensions of our images is huge, we can get a subsample of the rows and columns. We can selects indeces randomly, or we can do it deterministicly, say, in each 50 pixels, pick the firt 5 pixels (jump 45 pixels at a time.) Here is how we can select deterministicly:


```python
h, w = load_image(img_pths[0]).shape
print(h, w)

def indexing(length, small_increment, large_increment):
    nested = [[j+i for j in range(small_increment)] for i in range(0, length, large_increment)]
    return([x for sublist in nested for x in sublist])
    
random_col_index = indexing(w, 5, 50)
print(random_col_index)

random_row_index = indexing(h, 5, 50)
print(random_row_index)

```

    520 696
    [  0   1   2   3   4  50  51  52  53  54 100 101 102 103 104 150 151 152
     153 154 200 201 202 203 204 250 251 252 253 254 300 301 302 303 304 350
     351 352 353 354 400 401 402 403 404 450 451 452 453 454 500 501 502 503
     504 550 551 552 553 554 600 601 602 603 604 650 651 652 653 654]
    [  0   1   2   3   4  50  51  52  53  54 100 101 102 103 104 150 151 152
     153 154 200 201 202 203 204 250 251 252 253 254 300 301 302 303 304 350
     351 352 353 354 400 401 402 403 404 450 451 452 453 454 500 501 502 503
     504]


Here we change the images to numpy arrays in a loop. In the meantime, we select the subsample of rows and columns and flatten the selected sub-matrix:

We get the filenames as well to use it later for labeling the images correctly:


```python
img_data_ = [load_image(img)[np.array(random_row_index),:][:, np.array(random_col_index)].flatten().reshape(1, -1) for img in img_pths]

img_filenames = np.array([img.name for img in img_pths]).reshape(-1, 1)
print(img_filenames.shape)

img_data = np.concatenate(img_data_, axis=0)
print(img_data.shape)
```

    (2400, 1)
    (2400, 3850)


This step is stacking the filenames and numpy arrays (of images) and convert it to a pandas and gives names to the columns:


```python
name_img_data = np.hstack((img_filenames, img_data))
name_img_data = pd.DataFrame(name_img_data)
name_img_data.columns = ["image_name"] + ["x{}".format(i) for i in range(img_data.shape[1])]
```

Here we read in a csv file which contains labels of images and some of other image specific features. One of the columns is the name of the files (including the suffices.) We then convert the data to a pandas dataframe.


```python
# image labels

train_label_ = []
with open('./train_label.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        train_label_ += [row]
        
train_label = pd.DataFrame(train_label_[1:])
train_label.columns = train_label_[0]
```

We merge the two dataframes, the numeric matrix and the label matrix (both saved in dataframes):


```python
labeled_img_data_ = pd.merge(train_label, name_img_data, on=["image_name"], left_index=True, right_index=True, how='inner');
```

The filename column is not needed anymore and we delete it:


```python
labeled_img_data = labeled_img_data_[list(labeled_img_data_)[1:]]
```


```python
list(labeled_img_data)[:10]
```




    ['count', 'blur', 'stain', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']



To validate our results, we need to keep a part of our data aside for validation/development purposes. 10% is enough here. 


```python
train, test= train_test_split(labeled_img_data, test_size=0.1, random_state=42)
```


```python
print(list(train.shape))
print(list(test.shape))
```

    [2160, 3853]
    [240, 3853]


Let's save our matrices in a folder so that in future we can import them instead of repeating the whole procedure above:


```python
path_save = "/home/mr/SSC_case_study/matrix_forms/"

train.to_csv(path_save + 'train.csv', sep=',', index=False)
test.to_csv(path_save + 'test.csv', sep=',', index=False)
```


```python

```


```python

```
