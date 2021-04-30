# MGC_classifier
This repository is an attempt to see if combining the heatmaps of MGC with a convolutional classifier allows us to directly classify relationships between variables without relying on pure observation



# Current State Of Progress

Currently the classifier network has been implemented, a base static dataset, and the `nn.Dataset` custom modules. 

## To Do

- Implement the training and validation loops
- Implement Adversarial network
- Integrate adversarial network, generator and classifier
  - Do a tuning run
  - Set on continuous training