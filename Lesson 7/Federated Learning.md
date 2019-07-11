# Federated Learning

Federated Learning is a technique for training machine learning models on data to which we do not have access. instead of bringing all the data into one machines in the Cloud, and training a model, we're going to bring that model to the data, train it locally wherever the data lives and merely upload model updates to a central server.

Example: Auto-complete on mobile phones.

Every once in a while, it'll actually do a bit of local training on your device, on your own text messages, and it will send a new model, a slightly smarter model up to the Cloud and then later you'll get an updated aggregation of everyone else's model that also went up to the Cloud, thus giving you a smarter model. This happens without anyone having to divulge the private informations contained in their phone.

## Use Cases

Use Case 1: Predictive maintenance.

Federated Learning could be used to try to predict when your car needs to go into maintenance ahead of time by having a model that actually lives within all cars that are on the road, studying when they actually start to break down. In theory, a model just comes down to your car, it learns how to predict when your car is going to require maintenance and it uploads that ability to predict.
An update to the model, backup to the Cloud.

Use Case 2: Wearable medical devices

A machine learning model which could help you optimize certain parts of your health, whether it's your diet for having a better sleep cycle, or how much you move during the day for accomplishing some sort of wake up.

A person is actually not generating enough training data to be able to train a model like this.
But, if collaboratively trained as single machine learning model with thousands or millions of other people, then everyone can benefit from this machine learning model without anyone having to upload their data to a central Cloud.


### Deployed Case: ad blocking or an auto-complete inside of mobile browsers.

Distributed Dataset - Can't aggregate the dataset to once place but we need to use it to train the model. 
Eg: Medical data has legal restrictions and cannot be accessed publicly by anyone.
Eg: Competitive dynamics - In the case of predictive maintainance, the automobile companies won't want to share their vehicle breakdown data centrally.

## Advantage of Federated Learning
1. Reduce the bandwidth cost uploading data from multiple devices to the cloud everytime we want to train the model centrally.
2. User data privacy is maintained and better user experience with smarter models.
