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
#ifndef RRCA_DISTRIBUTIONEMBEDDING_CONDITIONALDISTRIBUTIONEMBEDDING_H_
#define RRCA_DISTRIBUTIONEMBEDDING_CONDITIONALDISTRIBUTIONEMBEDDING_H_


// #include<eigen3/Eigen/Dense>
namespace RRCA
{
namespace DISTRIBUTIONEMBEDDING
{
    
//     this is the traditionalconditional distribution embedding 
template<typename KernelMatrix, typename LowRank, typename KernelBasis,typename KernelMatrixY = KernelMatrix, typename LowRankY = LowRank, typename KernelBasisY = KernelBasis>
class ConditionalDistributionEmbedding
{


public:
    ConditionalDistributionEmbedding ( const Matrix& xdata_, const Matrix& ydata_) :
        xdata ( xdata_ ),
        ydata ( ydata_ ),
        Kx ( xdata_ ),
        Ky ( ydata_ ),
        basx(Kx,pivx),
        basy(Ky,pivy)
    {


    }
    const Matrix& getH() const {
        return(H);
    }
    
    unsigned int getSubspaceDimension() const {
        return(LX.cols() + LY.cols());
    } 
    
      /*
   *    \brief solves the low-rank problem unconstrained 
   */
        int solveUnconstrained ( double l1, double l2,double prec,double lam )
    {
        
        precomputeKernelMatrices ( l1, l2,prec,lam );
        h = (Qy * ( prob_vec.cwiseQuotient (prob_quadFormMat).reshaped(Qy.cols(), Qx.cols()) ) * Qx.transpose()).reshaped();
        H = h.reshaped(Qy.cols(), Qx.cols());

        return ( EXIT_SUCCESS );
    }
 
    
        /*
   *    \brief returns the vector the inner product of which with function evaluations gives the conditional expectaion
   */
    Matrix condExpfVec ( const Matrix& Xs ) const
    {
        const Matrix& Kxmultsmall  = basx.eval(Xs).transpose();

        return H * Kxmultsmall;
    }
    
    const iVector& getYPivots() const {
        return(pivy.pivots());
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
//     Matrix Kyblock_m; // the kernelfunctions of the important X ones accrdog to tensor prod rule
//     Matrix Kxblock_m; // the kernelfunctions of the important X ones accrdog to tensor prod rule

    Matrix C_Under; // the important ones
    Matrix C_Over; // the important ones

    Matrix Qx; // the basis transformation matrix
    Matrix Qy; // the basis transformation matrix
//     Matrix Q;
    
    Vector prob_quadFormMat; // this is a vector, because we only need the diagonal
    Vector prob_vec; // the important ones
    
    
    Vector xbound_min;
    Vector xbound_max;
    
    Vector ybound_min;
    Vector ybound_max;
    
    Matrix Qy_pos; // the important ones
    Matrix Qy_neg; // the important ones

    KernelMatrix Kx;
    KernelMatrixY Ky;

    LowRank pivx;
    LowRankY pivy;
    
    KernelBasis basx;
    KernelBasisY basy;
    

    /*
   *    \brief computes the kernel basis and tensorizes it
   */ 
    void precomputeKernelMatrices ( double l1, double l2,double prec, double lam )
    {
                Kx.kernel().l = l1;
        Ky.kernel().l = l2;

        // pivx.compute ( Kx,prec,0,RRCA_LOWRANK_STEPLIM  );
        // pivy.compute ( Ky,prec,0,RRCA_LOWRANK_STEPLIM  );   
        
        pivx.compute ( Kx,prec);
        pivy.compute ( Ky,prec); 
        
        // pivy.computeBiorthogonalBasis();
        
        basx.init(Kx, pivx.pivots());
        basx.initSpectralBasisWeights(pivx);

        
        
        Qx =  basx.matrixQ() ;
        Kxblock = basx.eval(xdata);
        
        

        LX = Kxblock * Qx;
        
        basy.init(Ky, pivy.pivots());
        basy.initSpectralBasisWeights(pivy);
        
        

        
        Qy =  basy.matrixQ() ;
        Kyblock = basy.eval(ydata);
        LY = Kyblock * Qy;
        
//         Qy_pos = Qy.cwiseMax(0.0);
//         Qy_neg = Qy.cwiseMin(0.0).cwiseAbs();
//         
//         xbound_min = LX.colwise().minCoeff().transpose();
//         xbound_max = LX.colwise().maxCoeff().transpose();
        
        precomputeHelper(lam);
    }
    

    
    
    
    void precomputeHelper(double lam){
        const unsigned int n ( LX.rows() );
        const unsigned int  rankx ( LX.cols() );
        const unsigned int  ranky ( LY.cols());
        const unsigned int  m ( rankx*ranky );
        
       
        prob_vec = (LY.transpose() * LX).reshaped();

        Xvar =  ( LX.transpose() * LX).diagonal();

        Matrix oida = Xvar.transpose().replicate(ranky,1);
        prob_quadFormMat= oida.reshaped().array()+n*lam;

    }
};
} // namespace DISTRIBUTIONEMBEDDING
}  // namespace RRCA
#endif
