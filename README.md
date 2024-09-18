# MeshNet

This project was done as a part of the senior project class at the **University of South Florida.** Working in a team of 4 students and alongside Machine Learning engineers at **Nielsen** our goal was to build a deep learning model to predict devices based on streaming data of a mesh network.

## Background

### Nielsen
Nielsen is a global measurement and data analytics company that provides the most complete and trusted view available of consumers and markets worldwide. Nielsen uses several tools and technologies to measure viewing habits of audiences. One of the key technologies that Nielsen employs is the **Nielsen Meter.** The meters are deployed on select households around the country and are connected to a home network. They record a variety of data, depending on the type of meter and its deployment.

### Mesh Networks
Traditionally, MAC addresses have been a reliable means of pinpointing specific devices in a network but with the introduction of **Mesh Networks**, the reliability of device identification has been compromised, leading to erroneous MAC addresses (incorrect or null devices) in the data the Nielsen Meter records in households that have a mesh network.

### Problem statement
Can we build an ML-based device-to-streaming mapping system that can predict the device (a.k.a MAC address) based on the streaming service data provided by the Nielsen meter with at least 80% prediction accuracy (rate of true positives).


