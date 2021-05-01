# MGC_classifier
This repository is an attempt to see if combining the heatmaps of MGC with a convolutional classifier allows us to directly classify relationships between variables without relying on pure observation



# Current State Of Progress

Currently the classifier network has been implemented, a base static dataset, and the `nn.Dataset` custom modules. 

The network, trained on the static dataset was able to achieve very good results, 

and as such, it is justified to try to implement a more complex data generation algorithm and adversarial network.

Results for an trained untuned run are shown below

![image-20210501163720095](C:\Users\walth\AppData\Roaming\Typora\typora-user-images\image-20210501163720095.png)

## Data

The data for the static dataset training is available at [Multiscale Graph Correlation Classification Data | Kaggle](https://www.kaggle.com/waltherwjohnson/multiscale-graph-correlation-classification-data)



## To Do

- Implement Adversarial network
- Integrate adversarial network, generator and classifier
  - Do a tuning run
  - Set on continuous training