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


covar = np.array([
    [1, -0.736924, -0.0826997],
    [-0.736924, 1, -0.562082],
    [-0.0826997, -0.562082, 1]
])


eigvals, eigvecs = np.linalg.eigh(covar)
# print(eigvals)
# print(eigvecs)

chol = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

# print(chol)

uniMatr = np.array([
    [0.459645,  0.0221783,  0.191117,  0.767538,  1.15418,  0.0820361, -1.77261,  -0.718471,  0.749513, -0.0504267],
    [0.189132, -0.233666, -0.443232,  1.30985,  -0.773168, -0.162637, -0.175362, -0.0670528, -0.209844, -0.456606],
    [-0.531634, 0.0140969, 0.0480184,  1.00209, -0.138363,  0.983943, 0.00739487, -2.01456,  -0.638118,  0.329326]
])

train = chol @ uniMatr + mean.reshape(-1, 1) #@ np.random.normal(size=(3, TRAIN)) + mean.reshape(-1, 1)
# print(train)

xtrain = train[1:3, :]
ytrain = train[0:1, :]

xtrain = np.ascontiguousarray(xtrain, dtype=np.float64)
ytrain = np.ascontiguousarray(ytrain, dtype=np.float64)


# print(xtrain)
# print(ytrain)

print("aaa")
emb = rrcapy.DistributionEmbedding(xtrain, ytrain)# init ok then data vanish
emb.printXdata()
emb.printKx()
print("bbb")
# print("\n\n")
# print(xtrain)

unimatrTest = np.array([
    [1.00204,   0.328421,   0.235561,   -2.44502, -0.0017024,   0.297616,  -0.395361,  -0.672721,    1.32543,  -0.333218],
    [1.45773,    0.13301,   0.238645,   0.280764,  -0.523022,  -0.432015,  -0.648212,  -0.773478,    1.11137,  -0.283469],
    [0.566143,   -1.94077,  -0.927159,    -1.2604,   0.468358,    0.51442,     1.6469,   -1.10256,   -1.04608,   -1.09322]
])

test = chol @unimatrTest + mean.reshape(-1, 1)#@ np.random.normal(size=(3, TEST)) + mean.reshape(-1, 1)
xtest = test[1:3, :]
ytest = test[0:1, :]

assert not np.isnan(xtrain).any(), "xtrain contains NaNs"
assert not np.isinf(xtrain).any(), "xtrain contains infinities"
assert not np.isnan(ytrain).any(), "ytrain contains NaNs"
assert not np.isinf(ytrain).any(), "ytrain contains infinities"


# emb.solveUnconstrained(l1, l2, tol, lam)
# print("unconstrained H:", emb.getH())

# condExpVecUC = emb.condExpfVec(xtest)
# condExpVecUCTrain = emb.condExpfVec(xtrain)

# print(condExpVecUC)
# print(condExpVecUCTrain)

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