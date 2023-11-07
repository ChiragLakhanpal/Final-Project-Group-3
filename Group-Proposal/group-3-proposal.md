# Group 3 Final Project Proposal
For our final project, we are creating a semantic segmentation network for the purpose of land cover classification in France using high-resolution satellite imagery data provided by the French National Institute of Geographical and Forest Information (IGN).

The dataset we will be using is called French Land cover from Aerospace ImageRy (FLAIR) and can be downloaded from the [official challenge website](https://ignf.github.io/FLAIR/).

The dataset contains over 20 billion pixels of high resolution satellite imagery, topographic data, and land cover annotations. We believe the size is more than sufficient for training a deep neural network. The satellite imagery consists of 77,412 tiles (images) that are 512x512 pixels each, and was captured at a 0.2m spatial resolution.

We intend to break the development of our network down into 3 types and compare all results so we choose the best network, defined by scoring the best metric against the test dataset. During the first step, we will create and train a custom multilayer Convolution network. In the next step, we will try to gain an advantage from transfer learning by integrating one or more baseline models into our network. We can experiment with fine-tuning these models for our task. Finally, time permitting, we will test the viability of a Vision Transformer network on this dataset. After gathering results from each step, we will determine the best network.

The training networks will be developed using the PyTorch framework. There are a number of reasons we’ve decided to go with this framework. PyTorch offers the best flexibility for developing complex models with access to the type of layers we’re interested in (Conv2D, Pool, VisionTranformer) and numerous loss optimization methods. PyTorch, similar to numpy, is also very modular which is beneficial for experimenting with different network architectures and patterns.

Working with satellite imagery data presents a unique challenge. Luckily, there are plenty of Python packages that can help with loading and processing this type of image data. Satellite imagery is typically stored as rasters in the GeoTIFF file format. Working with and manipulating raster data requires using packages like `rasterio` and `rioxarray`. Along with domain knowledge, we will also need to focus on reading the PyTorch documentation to understand how to develop a functional model architecture using convolution layers. [Semantic Segmentation](https://paperswithcode.com/task/semantic-segmentation) from Paperwithcode is a key resource for understanding which SOTA models are available for fine-tuning.

The performance of our networks will be judged on the mean Intersection-over-Union (mIoU) metric. This metric is calculated by comparing the ground truth bounding box to the predicted bounding box from the model. It is defined as the area of overlap between the bounding boxes and the total area (area of union) of those same bounding boxes. mIou is the average of the per-class IoUs computed for each test batch. We will also visualize bounding box results for a subset of images for illustrative purposes.

## TODO: create rough schedule
