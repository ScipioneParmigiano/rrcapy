////////////////////////////////////////////////////////////////////////////////
// This file is part of RRCA, the Roadrunner Covariance Analsys package.      //
//                                                                            //
// Copyright (c) 2021, Michael Multerer and Paul Schneider                    //
//                                                                            //
// All rights reserved.                                                       //
//                                                                            //
// This source code is subject to the BSD 3-clause license and without        //
// any warranty, see <https://github.com/muchip/RRCA> for further             //
// information.                                                               //
////////////////////////////////////////////////////////////////////////////////
#ifndef RRCA_KERNELMATRIX_KERNELMATRIX_H_
#define RRCA_KERNELMATRIX_KERNELMATRIX_H_
#include <iostream>

namespace RRCA {
template <typename KernelFunction, typename Derived>
struct KernelMatrix {
  typedef typename Derived::value_type value_type;

  typedef KernelFunction kernelfunction;
  //////////////////////////////////////////////////////////////////////////////
  KernelMatrix(const Eigen::EigenBase<Derived> &pts)
      : pts_(pts.derived()), dim_(pts.cols()){};
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Index cols() const { return dim_; }
  Eigen::Index rows() const { return dim_; }
  //////////////////////////////////////////////////////////////////////////////
  const KernelFunction &kernel() const { return kernel_; }
  KernelFunction &kernel() { return kernel_; }
  //////////////////////////////////////////////////////////////////////////////
  const Derived &pts() const { std::cout << 71; return pts_; std::cout << 72; };
  //////////////////////////////////////////////////////////////////////////////
  double operator()(int i, int j) const {
    // std::cout << 61 <<std::endl;
    return kernel_(pts_.col(i), pts_.col(j));
  }
  //////////////////////////////////////////////////////////////////////////////
  Vector col(Eigen::Index j) const {
    Vector retval(dim_);
    // std::cout << 51 << std::endl;
    // std::cout << "dim: " << dim_ << std::endl;
    for (auto i = 0; i < dim_; ++i)  {/*std::cout << "index col: "<< i << std::endl; std::cout<< operator()(i, j) << std::endl << "fallimento qui a asx?"<< std::endl; */retval(i) =operator()(i, j);};
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  Vector diagonal() const {
    Vector retval(dim_);
    for (auto i = 0; i < dim_; ++i) retval(i) = operator()(i, i);
    return retval;
  }
  Matrix full() const {
    std::cout << 41 << std::endl;
    Matrix retval(dim_, dim_);
    std::cout << 42 << std::endl;
    std::cout << "cols: " << retval.cols()<< std::endl;
    for (auto i = 0; i < retval.cols(); ++i) {/*std::cout << "retval: " <<retval.col(i) << std::endl; std::cout << "i" << std::endl; std::cout << "col: " <<col(i) << std::endl; */retval.col(i) = col(i);};
    std::cout << 43 << std::endl;
    return retval;
  }
  //////////////////////////////////////////////////////////////////////////////
  KernelFunction kernel_;
  const Derived &pts_;
  Eigen::Index dim_;
};

}  // namespace RRCA
#endif
