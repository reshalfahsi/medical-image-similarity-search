# Medical Image Similarity Search Using a Siamese Network With a Contrastive Loss


<div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/medical-image-similarity-search/blob/master/Medical_Image_Similarity_Search.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
</div>


Obtaining the ontological account of an image numerically can be earned via a Siamese network. The anatomy of this network has a twin architecture, consisting of convolutional and fully connected layers with shared weights. Each architecture digests an image and yields the vector embedding (the ontological or latent representation) of that image. These two vectors are then subjected to the Euclidean distance calculation. Next, the result is funneled to the last fully connected layer to get the logit describing their similarity. To learn the representation, here, we can leverage contrastive loss as our objective function to be optimized. The network is trained on paired images, i.e., positive and negative. In this project, the positive pairs are two images that belong to the same dataset, and the negative pairs are two images from distinct datasets. Here, subsets of the MedMNIST dataset are utilized: DermaMNIST, PneumoniaMNIST, RetinaMNIST, and BreastMNIST. Then, accuracy is used to evaluate the trained network. Afterward, we encode all images of the train and validation sets into embedding vectors and store them in the PostgreSQL database. So, sometimes later, we can use the embedding vectors to retrieve similar images based on a query image (we can obtain it from the test set). To find similar images, FAISS (Facebook AI Similarity Search) is employed. FAISS helps us seek the closest vectors to the query vector.


## Experiment

This [notebook](https://github.com/reshalfahsi/medical-image-similarity-search/blob/master/Medical_Image_Similarity_Search.ipynb) shows the overall code of this project.


## Result

## Quantitative Result

The loss and accuracy of the test set are exhibited quantitatively below:

Test Metric  | Score
------------ | -------------
Loss         | 0.012
Accuracy     | 98.58%


## Accuracy and Loss Curve

<p align="center"> <img src="https://github.com/reshalfahsi/medical-image-similarity-search/blob/master/assets/loss_curve.png" alt="loss_curve" > <br /> The loss curve on the train and validation sets of the Siamese network. </p>

<p align="center"> <img src="https://github.com/reshalfahsi/medical-image-similarity-search/blob/master/assets/acc_curve.png" alt="acc_curve" > <br /> The accuracy curve on the train and validation sets of the Siamese network. </p>


## Qualitative Result

The following picture presents four similar images contingent upon a query image of four different datasets, i.e., DermaMNIST, PneumoniaMNIST, RetinaMNIST, and BreastMNIST.

<p align="center"> <img src="https://github.com/reshalfahsi/medical-image-similarity-search/blob/master/assets/qualitative.png" alt="qualitative" > <br /> The image similarity search results for DermaMNIST (first row), PneumoniaMNIST (second row), RetinaMNIST (third row), and BreastMNIST (fourth row). </p>


## Credit

- [Image Similarity Estimation Using a Siamese Network With a Contrastive Loss](https://keras.io/examples/vision/siamese_contrastive/)
- [Near-Duplicate Image Search](https://keras.io/examples/vision/near_dup_search/)
- [Signature Verification using a "Siamese" Time Delay Neural Network](https://proceedings.neurips.cc/paper/1993/file/288cc0ff022877bd3df94bc9360b9c5d-Paper.pdf)
- [Neural Networks for Fingerprint Recognition](https://ieeexplore.ieee.org/document/6797067)
- [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
- [Learning a Similarity Metric Discriminatively, with Application to Face Verification](https://ieeexplore.ieee.org/document/1467314)
- [Python-Colab-Postgres](https://github.com/skupriienko/Python-Colab-Postgres)
- [Leveraging Google Colab to run Postgres: A Comprehensive Guide](https://dev.to/0xog_pg/leveraging-google-colab-to-run-postgres-a-comprehensive-guide-3kpn)
- [Using PostgreSQL in Python](https://www.datacamp.com/tutorial/tutorial-postgresql-python)
- [Introduction to Facebook AI Similarity Search (Faiss)](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/)
- [MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification](https://medmnist.com/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
