import sys
sys.path.append('../rrcapy')

import rrcapy
# print(dir(rrcapy))


import numpy as np
from rrcapy import DistributionEmbedding
import time

# Generate random data as in the C++ example
np.random.seed(int(time.time()))
TRAIN, TEST = 100, 100
mean = np.array([0, 0, 0])
rhoXY, rhoXZ, rhoYZ = np.random.uniform(-1.0, 1.0, 3)
sigX, sigY, sigZ = 1, 1, 1

covar = np.array([[sigX**2, sigX*sigY*rhoXY, sigX*sigZ*rhoXZ],
                  [sigX*sigY*rhoXY, sigY**2, sigY*sigZ*rhoYZ],
                  [sigX*sigZ*rhoXZ, sigY*sigZ*rhoYZ, sigZ**2]])

eigvals, eigvecs = np.linalg.eigh(covar)
eigvals = np.clip(eigvals, a_min=0, a_max=None)
chol = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

train = chol @ np.random.normal(size=(3, TRAIN))
train += mean[:, None]

xtrain = train[1:3, :]
ytrain = train[0:1, :]

# Instantiate the DistributionEmbedding object
embedding = DistributionEmbedding(xtrain, ytrain)

# Define parameters for the solveUnconstrained function
l1, l2, tol, lam = 0.8, 0.7, 0.001, 0.01

# Call solveUnconstrained
embedding.solveUnconstrained(l1, l2, tol, lam)

# # Check the resulting positiveOnGrid
# print("unconstrained positive on grid", embedding.positiveOnGrid())

# # Generate test data similarly
# test = chol @ np.random.normal(size=(3, TEST))
# test += mean[:, None]

# xtest = test[1:3, :]
# ytest = test[0:1, :]

# # Compute the conditional expectation vector
# condExpVecUC = embedding.condExpfVec(xtest)
# print("Unconstrained condExpVecUC", condExpVecUC)

# # Output results similar to the C++ code
# myfuns = [
#     lambda arg: 1.0 if arg > 2.0 else 0.0,
#     lambda arg: 1.0,
#     lambda arg: arg
# ]

# cond1 = np.apply_along_axis(myfuns[0], 0, ytrain)
# cond2 = np.apply_along_axis(myfuns[1], 0, ytrain)
# cond3 = np.apply_along_axis(myfuns[2], 0, ytrain)

# # Perform checks
# print("---- prob ----")
# print("unconstrained low rank:", cond1 @ condExpVecUC)
# print("---- exp  ----")
# print("unconstrained low rank:", cond3 @ condExpVecUC)

# # With training data
# condExpVecUCTrain = embedding.condExpfVec(xtrain)
# print("---- prob ---- with training data")
# print("unconstrained low rank:", cond1 @ condExpVecUCTrain)
# print("---- exp  ---- with training data")
# print("unconstrained low rank:", cond3 @ condExpVecUCTrain)
