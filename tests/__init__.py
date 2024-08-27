import sys
import numpy as np
sys.path.append('../rrcapy')

import rrcapy

# print(dir(rrcapy.DistributionEmbedding))

TRAIN = 100
TEST = 100
l1 = 0.8
l2 = 0.7
tol = 0.001
lam = 0.01

mean = np.array([0, 0, 0])

correldis = np.random.default_rng(1).uniform(-1.0, 1.0)
rhoXY = correldis
rhoXZ = correldis
rhoYZ = correldis
sigX = 1
sigY = 1
sigZ = 1

covar = np.array([[sigX * sigX, sigX * sigY * rhoXY, sigX * sigZ * rhoXZ],
                  [sigX * sigY * rhoXY, sigY * sigY, sigY * sigZ * rhoYZ],
                  [sigX * sigZ * rhoXZ, sigY * sigZ * rhoYZ, sigZ * sigZ]])

eigvals, eigvecs = np.linalg.eigh(covar)
chol = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

train = chol @ np.random.normal(size=(3, TRAIN)) + mean.reshape(-1, 1)
xtrain = train[1:3, :]
ytrain = train[0:1, :]


emb = rrcapy.DistributionEmbedding(xtrain, ytrain)

emb.solveUnconstrained(l1, l2, tol, lam)
# print("unconstrained H:", emb.getH())

# test = chol @ np.random.normal(size=(3, TEST)) + mean.reshape(-1, 1)
# xtest = test[1:3, :]
# ytest = test[0:1, :]

# condExpVecUC = emb.condExpfVec(xtest)
# condExpVecUCTrain = emb.condExpfVec(xtrain)

# def fun1(arg): return 1.0 if arg > 2.0 else 0.0
# def fun2(arg): return 1.0
# def fun3(arg): return arg

# cond1 = np.vectorize(fun1)(ytrain)
# cond2 = np.vectorize(fun2)(ytrain)
# cond3 = np.vectorize(fun3)(ytrain)

# print("Conditional Expectation Vector for Test Data:", condExpVecUC)
# print("Conditional Expectation Vector for Training Data:", condExpVecUCTrain)

# old error:
# python3: /usr/local/include/Eigen/src/Core/Redux.h:411: typename Eigen::internal::traits<T>::Scalar Eigen::DenseBase<Derived>::redux(const Func&) const [with BinaryOp = Eigen::internal::scalar_max_op<double, double, 0>; Derived = Eigen::CwiseUnaryOp<Eigen::internal::scalar_abs_op<double>, const Eigen::Matrix<double, -1, -1> >; typename Eigen::internal::traits<T>::Scalar = double]: Assertion `this->rows()>0 && this->cols()>0 && "you are using an empty matrix"' failed.
# Aborted (core dumped)

# new 
# python3: /usr/local/include/Eigen/src/Core/Block.h:121: Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>::Block(XprType&, Eigen::Index) [with XprType = const Eigen::Matrix<double, -1, -1>; int BlockRows = -1; int BlockCols = 1; bool InnerPanel = true; Eigen::Index = long int]: Assertion `(i>=0) && ( ((BlockRows==1) && (BlockCols==XprType::ColsAtCompileTime) && i<xpr.rows()) ||((BlockRows==XprType::RowsAtCompileTime) && (BlockCols==1) && i<xpr.cols()))' failed.
# Aborted (core dumped)