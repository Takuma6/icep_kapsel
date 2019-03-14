#ifndef LINEAR_OPERATOR_HPP
#define LINEAR_OPERATOR_HPP
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

/*
  http://www.interactive-graphics.de/SPlisHSPlasH/doc/html/_matrix_free_solver_8h_source.html
  https://spectralib.org/
  https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
  https://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html
  https://github.com/martinjrobins/Aboria
  https://martinjrobins.github.io/Aboria/Aboria/MatrixReplacement.html
*/
class MatrixReplacement;

namespace Eigen{
  namespace internal{
      template<>
      struct traits<MatrixReplacement> : public Eigen::internal::traits< Eigen::SparseMatrix<double> > {};
  }//namespace internal
}

class MatrixReplacement : public Eigen::EigenBase< MatrixReplacement >{
public:
    //types
    typedef double Scalar;
    typedef double RealScalar;
    typedef int    StorageIndex;
    typedef void(*MatrixVectorProductFunction)(const Scalar*, Scalar *);
    enum {
      ColsAtCompileTime    = Eigen::Dynamic,
      MaxColsAtCompileTime = Eigen::Dynamic,
      IsRowMajor           = false
    };

    Index rows() const {return num;}
    Index cols() const {return num;}

    template<typename Rhs>
    Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
      return Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
    }

    MatrixReplacement(): num(0), mv_function(NULL){}
    MatrixReplacement(unsigned int _num, MatrixVectorProductFunction _fp):
                    num(_num), mv_function(_fp){}
    void attachMatrix(unsigned int _num, MatrixVectorProductFunction _fp){
      num         = _num;
      mv_function = _fp;
    }
    MatrixVectorProductFunction getMVFunction(){return mv_function;}
  private:
    Index num;
    MatrixVectorProductFunction mv_function;
};

// Implementation of MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen{
  namespace internal{
    template<typename Rhs>
    struct generic_product_impl<MatrixReplacement, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
          : generic_product_impl_base<MatrixReplacement,Rhs,generic_product_impl<MatrixReplacement,Rhs> >
    {
      typedef typename Product<MatrixReplacement,Rhs>::Scalar Scalar;

      template<typename Dest>
      static void scaleAndAddTo(Dest& dst, const MatrixReplacement& lhs, const Rhs& rhs, const Scalar& alpha)
      {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        assert(alpha==Scalar(1) && "scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);
        // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
        // but let's do something fancier (and less efficient):
        const Scalar *vec = &rhs(0);
        Scalar       *res = &dst(0);
        MatrixReplacement& lhs_ = const_cast<MatrixReplacement&>(lhs);
        lhs_.getMVFunction()(vec, res);
      }
    };    // Product
  }// internal namespace
}// eigen namespace

#endif
