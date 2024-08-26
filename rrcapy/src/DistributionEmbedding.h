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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDING_H_
#define RRCA_DISTRIBUTIONEMBEDDING_DISTRIBUTIONEMBEDDING_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <set>
#include <vector>
#include <ctime>
#include <map>
#include <random>
#include <chrono>
#include <iostream>

#include "util/Macros.h"
#include "util/Multinomial.h"
#include "util/MultiIndexSet.h"
#include "KernelMatrix/Kernels.h"
#include "KernelMatrix/KernelMatrix.h"
#include "KernelBasis/KernelBasis.h"
#include "CholeskyDecomposition/CholeskyDecompositionBase.h"
#include "CholeskyDecomposition/PivotedCholeskyDecomposition.h"
#include "CholeskyDecomposition/NystromCholeskyDecomposition.h"


namespace RRCA{
    namespace DISTRIBUTIONEMBEDDING{
        template<typename KernelMatrix, typename LowRank, typename KernelBasis,typename KernelMatrixY = KernelMatrix, typename LowRankY = LowRank, typename KernelBasisY = KernelBasis>
        class DistributionEmbedding{
        typedef typename KernelMatrix::value_type value_type;



        public:
            DistributionEmbedding ( const Matrix& xdata_, const Matrix& ydata_) :
                xdata ( xdata_ ),
                ydata ( ydata_ ),
                Kx ( xdata_ ),
                Ky ( ydata_ ),
                basx(Kx,pivx),
                basy(Ky,pivy)
            {}


            /*
            *    \brief solves the unconstrained finite-dimensional optimization problem for given kernel parameterization
            */
                int solveUnconstrained ( double l1, double l2,double prec,double lam ){

                precomputeKernelMatrices ( l1, l2,prec,lam );
                std::cout << 3 << std::endl;

                h = (Qy * ( -prob_vec.cwiseQuotient (prob_quadFormMat).reshaped(Qy.cols(), Qx.cols()) ) * Qx.transpose()).reshaped();
                    std::cout << 4 << std::endl;


                H = h.reshaped(Qy.cols(), Qx.cols());
                std::cout << 5 << std::endl;


                return ( EXIT_SUCCESS );
            }

            /*
            *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
            */
            Matrix condExpfVec ( const Matrix& Xs, bool structuralYes = false ) const{

                const Matrix res = Kyblock *H*basx.eval(Xs).transpose() + Matrix::Constant(Kyblock.rows(),Xs.cols(),1);
                if(!structuralYes){
                    return(res.array().rowwise()/res.colwise().sum().array());
                }
                // const Matrix resres = res.array().rowwise()/res.colwise().sum().array();

                return ( res.array().cwiseMax(0.0).rowwise()/res.cwiseMax(0.0).colwise().sum().array());
            }



        private:
            const Matrix& xdata;
            const Matrix& ydata;

            Vector h; // the vector of coefficients
            Matrix H; // the matrix of coefficients, h=vec H

            Matrix LX; // the important ones
            Matrix LY; // the important ones

            Vector Xvar; // the sample matrix of the important X ones
            Vector Yvar; // the sample matrix of the important X ones

            Matrix Kxblock; // the kernelfunctions of the important X ones
            Matrix Kyblock; // the kernelfunctions of the important X ones

            Matrix Qx; // the basis transformation matrix
            Matrix Qy; // the basis transformation matrix

            Vector prob_quadFormMat; // this is a vector, because we only need the diagonal
            Vector prob_vec; // the important ones


            Matrix C_Under; // the important ones
            Matrix C_Over; // the important ones



            KernelMatrix Kx;
            KernelMatrixY Ky;

            LowRank pivx;
            LowRankY pivy;

            KernelBasis basx;
            KernelBasisY basy;

            //  spectral basis bounds. in the first row the mins and in the second row the maxs
            Vector xbound_min;
            Vector xbound_max;

            Vector ybound_min;
            Vector ybound_max;

            //     double tol;
            static constexpr double p = 1.0;

            /*
            *    \brief computes the kernel basis and tensorizes it
            */ 
            void precomputeKernelMatrices ( double l1, double l2,double prec, double lam ){
            //         RRCA::IO::Stopwatch sw;
            //         sw.tic();

                Kx.kernel().l = l1;
                Ky.kernel().l = l2;
            
                pivx.compute ( Kx,prec);
                std::cout << 1 << std::endl;
                basx.init(Kx, pivx.pivots());
                
                std::cout << 1 << std::endl;
                basx.initSpectralBasisWeights(pivx);
                
                
                Qx =  basx.matrixQ() ;
                Kxblock = basx.eval(xdata);
                

                LX = Kxblock * Qx;
                
                            std::cout << 2 << std::endl;

                pivy.compute ( Ky,prec);
                basy.init(Ky, pivy.pivots());
                basy.initSpectralBasisWeights(pivy);
                
            std::cout << 4 << std::endl;

                
                Qy =  basy.matrixQ() ;
                Kyblock = basy.eval(ydata);
                
                            std::cout << 5 << std::endl;


                LY = Kyblock * Qy;
                
                xbound_min = LX.colwise().minCoeff().transpose();
                xbound_max = LX.colwise().maxCoeff().transpose();
                
                            std::cout << 6 << std::endl;

                
                ybound_min = LY.colwise().minCoeff().transpose();
                ybound_max = LY.colwise().maxCoeff().transpose();
                
                            std::cout << 7 << std::endl;

            //      bounds
                C_Under = ybound_min.cwiseMax(0) * xbound_min.cwiseMax(0).transpose()+ybound_max.cwiseMin(0).cwiseAbs() * xbound_max.cwiseMin(0).cwiseAbs().transpose() 
                        - (ybound_max.cwiseMax(0) * xbound_min.cwiseMin(0).cwiseAbs().transpose()).cwiseMax(ybound_min.cwiseMin(0).cwiseAbs() * xbound_max.cwiseMax(0).transpose());
                        
                C_Over = (ybound_max.cwiseMax(0) * xbound_max.cwiseMax(0).transpose()).cwiseMax(ybound_min.cwiseMin(0).cwiseAbs() * xbound_min.cwiseMin(0).cwiseAbs().transpose()) 
                        - ybound_max.cwiseMin(0).cwiseAbs() * xbound_min.cwiseMax(0).transpose()-ybound_min.cwiseMax(0) * xbound_max.cwiseMin(0).cwiseAbs().transpose();
                
                            std::cout << 8 << std::endl;

                

                precomputeHelper(lam);

                            std::cout << 9 << std::endl;

                
                // std::cout << " passed by precompute " << LX.cols() << '\t' << LY.cols() << std::endl;

            }

            void precomputeHelper(double lam){
                const unsigned int n ( LX.rows() );
                const unsigned int  rankx ( LX.cols() );
                const unsigned int  ranky ( LY.cols());
                // const unsigned int  m ( rankx*ranky );
                
                Vector p_ex_k = (LY.transpose() * LX).reshaped();
                Vector px_o_py_k = (LY.colwise().mean().transpose() * LX.colwise().mean()).reshaped()*n;

                Xvar =  ( LX.transpose() * LX).diagonal()  ;
                Yvar =  ( LY.transpose() * LY).diagonal() ;

                prob_quadFormMat= (Yvar * Xvar.transpose()).reshaped().array()/static_cast<double> ( n )+n*lam;
                prob_vec = px_o_py_k-p_ex_k;
            }
        };


    } // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif
