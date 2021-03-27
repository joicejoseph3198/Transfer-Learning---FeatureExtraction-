# Transfer Learning - FeatureExtraction 
Treating networks as feature extractors, forward propagating images until a given layer, and then taking these activations and treating them as feature vectors. Using these feature vectors, we train an off-the-shelf machine learning model such a Linear SVM, Logistic Regression classifier, or Random Forest on top of these features to obtain a classifier that recognizes new classes of images.

### Table of Contents

- [Description](#description)
- [Working](#working)
- [Results](#results)


---

## Description


Deep neural networks trained on large-scale datasets such as ImageNet have demonstrated to be excellent at the task of transfer learning. These networks learn a set of rich, discriminating features to recognize 1,000 separate object classes. It makes sense that these filters can be reused for classification tasks other than what the CNN was originally trained on. There is no rule that says we must allow the image to forward propagate through the entire network. Instead, we can stop the propagation at an arbitrary layer, such as an activation or pooling layer, extract the values from the network at this time, and then use them as feature vectors.



[Back To The Top](#read-me-template)

---

## Working

#### Installation
The requirements document provided in the repository has the list of all the packages that are needed for implementing this project.

```html
    pip install -r requirements.txt 
```
* hdf5datasetwriter. py :  We define a Python class named HDF5DatasetWriter, which as the name suggests, is responsible for taking an input set of NumPy arrays and writing them to HDF5 format. 
    * The dims parameter controls the dimension or shape of the data we will be storing in the dataset. 
    * In the context of transfer learning and feature extraction, we’ll be using the VGG16 architecture and taking the outputs after the final POOL layer. The output of the final POOL layer is 512x7x7 which, when flattened, yields a feature vector of length 25,088. Therefore, when using VGG16 for feature extraction, we’ll set dims=(N, 25088) where N is the total number of images in our Database. 
    * outputPath – this is the path to where our output HDF5 file will be stored on disk. 
    * The optional dataKey is the name of the dataset that will store the data our algorithm will learn from.
    * bufSize controls the size of our in-memory buffer, which we default to 1,000 feature vectors/images. Once we reach bufSize, we’ll flush the buffer to the HDF5 dataset

* extract_features.py: 
    * We import the Keras implementation of the pre-trained VGG16 network – this is the architecture we’ll be using as our feature extractor.
    * The extract_features.py script will require two command line arguments, followed by two optional ones. The --dataset switch controls the path to our input directory of images that we wish to extract features from. The --output switch determines the path to our output HDF5 data file. We can then supply a --batch-size – this is the number of images in a batch that will be passed through VGG16 at a time. A value of 32 is reasonable here, but you can increase it if your machine has sufficient memory. The --buffer-size switch controls the number of extracted features we’ll store in memory before writing the buffer.
    * We load the pre-trained VGG16 network from disk; and include the parameter include_top=False – supplying this value indicates that the final fully connected layers should not be included in the architecture. Therefore, when forward propagating an image through the network, we’ll obtain the feature values after the final POOL layer.
    * Next we start looping over our imagePaths in batches of --batch-size and extract the image paths and labels for the corresponding batch, and initialize a list to store the images about to be loaded and fed into VGG16.
    * We obtain our feature vectors for the images in batchImages, after we call the .predict method of model. 
    * To treat these values as a feature vector, we  flatten them into an array with shape (N, 25088) and store it in HDF5 format.

The first dataset we extract features from using VGG16 is the “Animals” dataset. This dataset consists of 3,000 images, of three classes: dogs, cats, and pandas. To utilize VGG16 to extract features from these images, simply execute the following command:

```html
    python extract_features.py --dataset datasets\animals --output datasets\animals\hdf5\features.hdf5
```
Next we extract features from using VGG16 is the "Flowers17" dataset.

```html
    python extract_features.py --dataset datasets\flowers17
    --output datasets/flowers17/hdf5/features.hdf5
```  
* train_model.py : 
    * To train a Logistic Regression classifier on the features extracted via the VGG16 network on the datasets (animals,  flowers17).
    * Running a grid search over the parameter C, the strictness  of the Logistic Regression classifier to determine what the optimal value is.
```html
    python train_model.py --db datasets\{dataset}\hdf5\features.hdf5 --model {dataset}.cpickle
```  

## Results

Networks such as VGG are capable of performing transfer learning, encoding their discriminative
features into output activations that we can use to train our own custom image classifiers

On Animal Dataset:

![Project Image1](https://raw.githubusercontent.com/joicejoseph3198/Images/main/featureextractionanimals.png)

On Flowers17 Dataset:

![Project Image2](https://raw.githubusercontent.com/joicejoseph3198/Images/main/featureextractionflower1.png)

![Project Image3](https://raw.githubusercontent.com/joicejoseph3198/Images/main/featureextractionflower2.png)

[Back To The Top](#read-me-template)