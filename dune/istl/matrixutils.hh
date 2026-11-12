// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_MATRIXUTILS_HH
#define DUNE_ISTL_MATRIXUTILS_HH

#include <set>
#include <vector>
#include <limits>
#include <numeric>
#include <dune/common/typetraits.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/dynmatrix.hh>
#include <dune/common/diagonalmatrix.hh>
#include <dune/common/scalarmatrixview.hh>
#include <dune/common/concepts/number.hh>
#include <dune/istl/scaledidmatrix.hh>
#include "istlexception.hh"

namespace Dune
{

#ifndef DOYXGEN
  template<typename B, typename A>
  class BCRSMatrix;


  template<typename K, int n, int m>
  class FieldMatrix;

  template<class T, class A>
  class Matrix;

namespace Experimental {
  template<typename B, typename I, typename A>
  class BCRSMatrix;

  template <typename B, typename P, typename A>
  class Matrix;
}

#endif

  /**
   * @addtogroup ISTL_SPMV
   * @{
   */
  /**
   * @file
   * @brief Some handy generic functions for ISTL matrices.
   * @author Markus Blatt
   */
  /**
   * @brief Check whether the a matrix has diagonal values
   * on blocklevel recursion levels.
   */
  template<class Matrix, std::size_t blocklevel, std::size_t l=blocklevel>
  struct CheckIfDiagonalPresent
  {
    /**
     * @brief Check whether the a matrix has diagonal values
     * on blocklevel recursion levels.
     */
    static void check([[maybe_unused]] const Matrix& mat)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      typedef typename Matrix::ConstRowIterator Row;
      typedef typename Matrix::ConstColIterator Entry;
      for(Row row = mat.begin(); row!=mat.end(); ++row) {
        Entry diagonal = row->find(row.index());
        if(diagonal==row->end())
          DUNE_THROW(ISTLError, "Missing diagonal value in row "<<row.index()
                                                                <<" at block recursion level "<<l-blocklevel);
        else{
          auto m = Impl::asMatrix(*diagonal);
          CheckIfDiagonalPresent<decltype(m),blocklevel-1,l>::check(m);
        }
      }
#endif
    }
  };

  template<class Matrix, std::size_t l>
  struct CheckIfDiagonalPresent<Matrix,0,l>
  {
    static void check(const Matrix& mat)
    {
      typedef typename Matrix::ConstRowIterator Row;
      for(Row row = mat.begin(); row!=mat.end(); ++row) {
        if(row->find(row.index())==row->end())
          DUNE_THROW(ISTLError, "Missing diagonal value in row "<<row.index()
                                                                <<" at block recursion level "<<l);
      }
    }
  };

  template<typename FirstRow, typename... Args>
  class MultiTypeBlockMatrix;

  template<std::size_t blocklevel, std::size_t l, typename T1, typename... Args>
  struct CheckIfDiagonalPresent<MultiTypeBlockMatrix<T1,Args...>,
      blocklevel,l>
  {
    typedef MultiTypeBlockMatrix<T1,Args...> Matrix;

    /**
     * @brief Check whether the a matrix has diagonal values
     * on blocklevel recursion levels.
     */
    static void check(const Matrix& /* mat */)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      // TODO Implement check
#endif
    }
  };

  /**
   * @brief Get the number of nonzero fields in the matrix.
   *
   * This is not the number of nonzero blocks, but the number of non
   * zero scalar entries (on blocklevel 1) if the matrix is viewed as
   * a flat matrix.
   *
   * For FieldMatrix this is simply the number of columns times the
   * number of rows, for a BCRSMatrix<FieldMatrix<K,n,m>> this is the
   * number of nonzero blocks time n*m.
   */
  template<class M>
  inline auto countNonZeros(const M&,
                            [[maybe_unused]] typename std::enable_if_t<Dune::IsNumber<M>::value>* sfinae = nullptr)
  {
    return 1;
  }

  template<class M>
  inline auto countNonZeros(const M& matrix,
                            [[maybe_unused]] typename std::enable_if_t<!Dune::IsNumber<M>::value>* sfinae = nullptr)
  {
    typename M::size_type nonZeros = 0;
    for(auto&& row : matrix)
      for(auto&& entry : row)
        nonZeros += countNonZeros(entry);
    return nonZeros;
  }

  /*
     template<class M>
     struct ProcessOnFieldsOfMatrix
   */

  /** @} */
  namespace
  {
    struct CompPair {
      template<class G,class M>
      bool operator()(const std::pair<G,M>& p1, const std::pair<G,M>& p2) const
      {
        return p1.first<p2.first;
      }
    };

  }
  template<class M, class C>
  void printGlobalSparseMatrix(const M& mat, C& ooc, std::ostream& os)
  {
    typedef typename C::ParallelIndexSet::const_iterator IIter;
    typedef typename C::OwnerSet OwnerSet;
    typedef typename C::ParallelIndexSet::GlobalIndex GlobalIndex;

    GlobalIndex gmax=0;

    for(IIter idx=ooc.indexSet().begin(), eidx=ooc.indexSet().end();
        idx!=eidx; ++idx)
      gmax=std::max(gmax,idx->global());

    gmax=ooc.communicator().max(gmax);
    ooc.buildGlobalLookup();

    for(IIter idx=ooc.indexSet().begin(), eidx=ooc.indexSet().end();
        idx!=eidx; ++idx) {
      if(OwnerSet::contains(idx->local().attribute()))
      {
        typedef typename  M::block_type Block;

        std::set<std::pair<GlobalIndex,Block>,CompPair> entries;

        // sort rows
        typedef typename M::ConstColIterator CIter;
        for(CIter c=mat[idx->local()].begin(), cend=mat[idx->local()].end();
            c!=cend; ++c) {
          const typename C::ParallelIndexSet::IndexPair* pair
            =ooc.globalLookup().pair(c.index());
          assert(pair);
          entries.insert(std::make_pair(pair->global(), *c));
        }

        //wait until its the rows turn.
        GlobalIndex rowidx = idx->global();
        GlobalIndex cur=std::numeric_limits<GlobalIndex>::max();
        while(cur!=rowidx)
          cur=ooc.communicator().min(rowidx);

        // print rows
        typedef typename std::set<std::pair<GlobalIndex,Block>,CompPair>::iterator SIter;
        for(SIter s=entries.begin(), send=entries.end(); s!=send; ++s)
          os<<idx->global()<<" "<<s->first<<" "<<s->second<<std::endl;


      }
    }

    ooc.freeGlobalLookup();
    // Wait until everybody is finished
    GlobalIndex cur=std::numeric_limits<GlobalIndex>::max();
    while(cur!=ooc.communicator().min(cur)) ;
  }

  //! Utility to get number of row and column entries of a matrix M, taking into account the block structure.
  template<typename M>
  struct MatrixDimension;

  //! Utility to get number of row and column entries of a matrix M, taking into account the block structure.
  template<Concept::Number M>
  struct MatrixDimension<M>
  {
    //! Get the number of numeric entries in the rows of the matrix A
    static std::size_t rowdim(const M& A)
    {
      return 1;
    }

    //! Get the number of numeric entries in the columns of the matrix A
    static std::size_t coldim(const M& A)
    {
      return 1;
    }

    //! Get the number of numeric entries in the i-th row of the matrix A
    template<typename U>
    static U rowdim(const M& A, U /*i*/)
    {
      return U(1);
    }

    //! Get the number of numeric entries in the c-th column of the matrix A
    template<typename U>
    static U coldim(const M& A, U /*c*/)
    {
      return U(1);
    }
  };

  //! Utility to get number of row and column entries of a matrix M, taking into account the block structure.
  template<typename M>
  requires (!Concept::Number<M> && Concept::Number<typename M::block_type>)
  struct MatrixDimension<M>
  {
  private:
    using size_type = typename M::size_type;
  public:
    //! Get the number of numeric entries in the rows of the matrix A
    static size_type rowdim(const M& A)
    {
      return A.N();
    }

    //! Get the number of numeric entries in the columns of the matrix A
    static size_type coldim(const M& A)
    {
      return A.M();
    }

    //! Get the number of numeric entries in the i-th row of the matrix A
    static size_type rowdim(const M& A, size_type i)
    {
      return 1;
    }

    //! Get the number of numeric entries in the c-th column of the matrix A
    static size_type coldim(const M& A, size_type c)
    {
      return 1;
    }
  };


  //! Utility to get number of row and column entries of a matrix M, taking into account the block structure.
  template<typename M>
  struct MatrixDimension
  {
  private:
    using block_type = typename M::block_type;
    using block_dim = MatrixDimension<block_type>;
    using size_type = typename M::size_type;
  public:
    //! Get the number of numeric entries in the rows of the matrix A
    static size_type rowdim(const M& A)
    {
      size_type nnz = 0;
      for (auto row=A.begin(); row!=A.end(); ++row)
        nnz += MatrixDimension<M>::rowdim(A, row.index());
      return nnz;
    }

    //! Get the number of numeric entries in the columns of the matrix A
    static size_type coldim(const M& A)
    {
      constexpr size_type ui_max = std::numeric_limits<size_type>::max();
      // The following code has a complexity of nnz, and
      // typically a very small constant.
      std::vector<size_type> coldims(A.M(), ui_max);

      for (auto row=A.begin(); row!=A.end(); ++row)
        for (auto col=row->begin(); col!=row->end(); ++col) {
          // only compute blocksizes we don't already have
          if (coldims[col.index()] == ui_max)
            coldims[col.index()] = block_dim::coldim(*col);
#ifdef DUNE_ISTL_WITH_CHECKING
          else if (coldims[col.index()] != block_dim::coldim(*col))
            DUNE_THROW(ISTLError, "Inconsistent block sizes in column "<<col.index());
#endif
        }

      auto handle_empty = [](size_type a, size_type b) {
        if (b == ui_max) {
          if (std::default_initializable<block_type>)
            return a + block_dim::coldim(block_type{});
          else
            DUNE_THROW(RangeError, "Matrix column is empty. It is not possible to determine its dimension dynamically.");
        }
        return a + b;
      };
      return std::accumulate(coldims.begin(), coldims.end(), size_type(0), handle_empty);
    }

    //! Get the number of numeric entries in the i-th row of the matrix A
    static size_type rowdim(const M& A, size_type i)
    {
      // find first valid column in i-th row
      if(auto row = A.begin() + i; row->begin() != row->end())
        return block_dim::rowdim(*row->begin());
      else if (std::default_initializable<block_type>)
        return block_dim::rowdim(block_type{});
      else
        DUNE_THROW(RangeError, "Matrix row is empty. It is not possible to determine its dimension dynamically.");
    }

    //! Get the number of numeric entries in the c-th column of the matrix A
    static size_type coldim(const M& A, size_type c)
    {
      // find first row with entry in column c
      for (auto rowIt = A.begin(); rowIt != A.end(); ++rowIt) {
        if (rowIt->size() == A.M()) {
          auto colIt = rowIt->begin() + c;
#ifdef DUNE_ISTL_WITH_CHECKING
          if(colIt.index() != c)
            DUNE_THROW(ISTLError, "Column iterator number " << c << " is the " << rowIt.index() << "-th rowIt->");
#endif
          return block_dim::coldim(*colIt);
        } else {
          for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
            if (colIt.index() == c)
              return block_dim::coldim(*colIt);
        }
      }
      // not found
      return 0;
    }
  };

  /**
   * @brief Test whether a type is an ISTL Matrix
   */
  template<typename T>
  struct IsMatrix
  {
    enum {
      /**
       * @brief True if T is an ISTL matrix
       */
      value = false
    };
  };

  template<typename T>
  struct IsMatrix<DenseMatrix<T> >
  {
    enum {
      /**
       * @brief True if T is an ISTL matrix
       */
      value = true
    };
  };


  template<typename T, typename A>
  struct IsMatrix<BCRSMatrix<T,A> >
  {
    enum {
      /**
       * @brief True if T is an ISTL matrix
       */
      value = true
    };
  };

  template<typename T, typename P, typename A>
  struct IsMatrix<Experimental::BCRSMatrix<T,P,A> >
  {
    enum {
      /**
       * @brief True if T is an ISTL matrix
       */
      value = true
    };
  };
}
#endif
