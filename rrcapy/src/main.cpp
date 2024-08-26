#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "DistributionEmbedding.h" 

namespace py = pybind11;

PYBIND11_MODULE(distribution_embedding, m) {
    py::class_<RRCA::DISTRIBUTIONEMBEDDING::DistributionEmbedding<
                RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>,
                RRCA::PivotedCholeskyDecompositon<
                    RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>,
                RRCA::KernelBasis<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>
                >
             >(m, "DistributionEmbedding")
        .def(py::init<const RRCA::Matrix&, const RRCA::Matrix&>())
        .def("solveUnconstrained", &RRCA::DISTRIBUTIONEMBEDDING::DistributionEmbedding<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>,
                RRCA::PivotedCholeskyDecompositon<
                    RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>,
                RRCA::KernelBasis<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>
                >::solveUnconstrained)
        .def("condExpfVec", &RRCA::DISTRIBUTIONEMBEDDING::DistributionEmbedding<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>,
                RRCA::PivotedCholeskyDecompositon<
                    RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>,
                RRCA::KernelBasis<RRCA::KernelMatrix<RRCA::GaussianKernel, RRCA::Matrix>>
                >::condExpfVec);
}
