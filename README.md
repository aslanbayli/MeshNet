# MeshNet

This project was done as a part of the senior project class at the **University of South Florida.** Working in a team of 4 students and alongside Machine Learning engineers at **Nielsen** our goal was to build a deep learning model to predict devices based on streaming data of a mesh network.

## Background

### Nielsen
<img src="https://github.com/user-attachments/assets/16d26bde-c426-4b44-8e49-ebee0a730957" width="500">

Nielsen is a global measurement and data analytics company that provides the most complete and trusted view available of consumers and markets worldwide. Nielsen uses several tools and technologies to measure viewing habits of audiences. One of the key technologies that Nielsen employs is the **Nielsen Meter.** The meters are deployed on select households around the country and are connected to a home network. They record a variety of data, depending on the type of meter and its deployment.

### Mesh Networks
![Mesh Animation](https://github.com/user-attachments/assets/7f0f55a7-ecdd-4f8d-8641-074046be623e)

Traditionally, MAC addresses have been a reliable means of pinpointing specific devices in a network but with the introduction of **Mesh Networks**, the reliability of device identification has been compromised, leading to erroneous MAC addresses (incorrect or null devices) in the data the Nielsen Meter records in households that have a mesh network.

### Problem statement
Can we build an ML-based device-to-streaming mapping system that can predict the device (a.k.a MAC address) based on the streaming service data provided by the Nielsen meter with at least 80% prediction accuracy (rate of true positives).

## Design
Because mesh networks naturally form a graph structured data, we have arrived at a conclusion to model our deep learning network as a Graph Neural Network (GNN).

### GNNS
A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-context) that preserves graph symmetries (permutation invariances).

<img src="https://github.com/user-attachments/assets/6880faa9-3935-41ca-b4fa-08ceffbf6872" width="800">

__source: Sanchez-Lengeling, B., Reif, E., Pearce, A., & Wiltschko, A. (2021). A gentle introduction to graph neural networks. Distill, 6(8). https://doi.org/10.23915/distill.00033__

### Our solution
<img src="https://github.com/user-attachments/assets/b9f2d561-8e59-4d07-a503-7ef822462770" width="500">    

In order to represent the mesh network data in the most optimal way, we have decided to introducde three types of nodes - devices, connections (streaming info.), and streaming events. The edges connecting the nodes include the likelihood of the event happening.  

<img src="https://github.com/user-attachments/assets/09c51657-f86c-47bc-8d5a-e4f66d507df5" width="500">

We use non-mesh household data as ground truth in order to train our network, and use mesh network data for inference purposes. 

## Results

![image](https://github.com/user-attachments/assets/a0f4d207-4cdb-4516-ad8d-ea1528c4143a)

As can be seen by the results, our solution was able to exceed the expected performance and successfully detect devices in a mesh network.

## Overcoming Project Challenges
**Choosing the Right Machine Learning Model**
- Complexity in model selection: accuracy, efficiency, scalability.
- Variety of ML architectures considered and evaluated.

**Data Preprocessing**
- Handling inconsistencies in data formats.
- Dealing with large volumes and diverse types of data.
- Ensuring integrity and usefulness of data post-processing.


## Key Learnings and Innovative Solutions
**Effective Data Normalization**
- Understanding the criticality of normalizing diverse data types.
- Techniques developed for effective normalization, ensuring data integrity for model input.

**Maintaining Pattern Recognition in Normalization**
- Importance of preserving patterns in bandwidth usage and log times during normalization.
- Innovative approaches to retain essential patterns for the model's accuracy.

## Future Directions and Potential Enhancements
**Exploring Alternative Machine Learning Models**
- Investigating other ML models to streamline and potentially enhance the solution.
- Evaluating models based on efficiency, accuracy, and their fit for the specific challenges of MAC address prediction in mesh networks.

**Acquiring Additional Nielsen Meter Data**
- The importance of a larger, more diverse dataset for refining model accuracy.
- Plans to access more extensive Nielsen Meter data, including different network types or more varied usage patterns.

