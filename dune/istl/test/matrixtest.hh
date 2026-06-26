// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_TEST_MATRIXTEST_HH
#define DUNE_ISTL_TEST_MATRIXTEST_HH

/** \file
 * \brief Infrastructure for testing the dune-istl matrix interface
 *
 * This file contains various methods that test parts of the dune-istl matrix interface.
 * They only test very general features, i.e., features that every dune-istl matrix should
 * have.
 *
 * At the same time, these tests should help to define what the dune-istl matrix interface
 * actually is.  They may not currently define the entire interface, but they are a lower
 * bound: any feature that is tested here is part of the dune-istl matrix interface.
 */

#include <dune/istl/blocklevel.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/foreach.hh>

#include <dune/common/exceptions.hh>
#include <dune/common/classname.hh>
#include <dune/common/float_cmp.hh>
#include <dune/common/scalarmatrixview.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/test/iteratortest.hh>

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <complex>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace Dune
{
  namespace MatrixTestImpl
  {

    template <class L, class R>
    auto relativeDifference(const L& lhs, const R& rhs)
    {
      using scalar_field = typename PromotionTraits<typename L::field_type, typename R::field_type>::PromotedType;
      using real_type = typename FieldTraits<scalar_field>::real_type;
      L diff = lhs;
      diff -= rhs;
      const real_type scale = std::max(real_type{1.0}, rhs.infinity_norm());
      return diff.infinity_norm() / scale;
    }

    template <class L, class R>
    void expectNear(const L& lhs, const R& rhs, double tol, const char* message)
    {
      const auto rel = relativeDifference(lhs, rhs);
      if (rel > tol)
        DUNE_THROW(ISTLError, message << " (relative difference " << rel << ", tolerance " << tol << ")");
    }

    template <class Matrix, class DomainVector, class RangeVector>
    void testMvDenseScalarMatrix(const Matrix& matrix,
                                    const DomainVector& domain,
                                    const RangeVector& range)
    {
      const std::size_t n = matrix.N();
      const std::size_t m = matrix.M();
      using value_type = typename Matrix::field_type;

      std::vector<std::vector<value_type>> dense(n, std::vector<value_type>(m, value_type(0)));

      for (auto rowIt = matrix.begin(); rowIt != matrix.end(); ++rowIt)
        for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt)
          dense[rowIt.index()][colIt.index()] = *colIt;

      const double tol = 1e-11;

      RangeVector yExpected(range);
      yExpected = 0;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
          yExpected[i] += dense[i][j] * domain[j];

      RangeVector yComputed(range);
      yComputed = 0;
      matrix.mv(domain, yComputed);
      expectNear(yComputed, yExpected, tol, "Dense check failed for mv()");

      DomainVector ytExpected(domain);
      ytExpected = 0;
      for (std::size_t j = 0; j < m; ++j)
        for (std::size_t i = 0; i < n; ++i)
          ytExpected[j] += dense[i][j] * range[i];

      DomainVector ytComputed(domain);
      ytComputed = 0;
      matrix.mtv(range, ytComputed);
      expectNear(ytComputed, ytExpected, tol, "Dense check failed for mtv()");
    }
  }

  /** \brief Test whether the given type is default- and copy-constructible */
  template <typename Matrix>
  void testMatrixConstructibility()
  {
    static_assert(std::is_default_constructible<Matrix>::value, "Matrix type is not default constructible");
    static_assert(std::is_copy_constructible<Matrix>::value, "Matrix type is not copy constructible");
  }

  template <typename Matrix, typename DomainVector, typename RangeVector>
  void testMatrixVectorProducts(const Matrix& matrix, const DomainVector& domain, const RangeVector& range)
  {
    using field_type = typename Matrix::field_type;
    using real_type = typename FieldTraits<field_type>::real_type;
    constexpr real_type tol = 1e-11;

    // Fill deterministic non-zero values
    std::size_t seed = 1;
    auto fillDeterministic = [&](auto& vec) {
      flatVectorForEach(vec, [&]<class T>(T& entry, auto i){
        entry = 0.125 * static_cast<T>((seed % 11) + 1);
        seed += 1;
      });
    };

    DomainVector x(domain);
    RangeVector z(range);
    fillDeterministic(x);
    fillDeterministic(z);

    RangeVector y0(range);
    DomainVector yt0(domain);
    fillDeterministic(y0);
    fillDeterministic(yt0);

    RangeVector ax(range);
    ax = 0;
    DomainVector atz(domain);
    atz = 0;
    DomainVector ahz(domain);
    ahz = 0;

    if constexpr (requires { typename Matrix::block_type; })
      if constexpr (IsNumber<typename Matrix::block_type>::value)
        if (matrix.N() * matrix.M() < 1000)
          MatrixTestImpl::testMvDenseScalarMatrix(matrix, x, z);

    std::cout << "Testing matrix-vector products on " << className<Matrix>() << std::endl;
    matrix.mv(x, ax);

    matrix.mtv(z, atz);
    matrix.umhv(z, ahz);

    // y += A x
    {
      RangeVector y = y0;
      RangeVector expected = y0;
      expected += ax;
      matrix.umv(x, y);
      MatrixTestImpl::expectNear(y, expected, tol, "umv() produced incorrect result");
    }

    // y -= A x
    {
      RangeVector y = y0;
      RangeVector expected = y0;
      expected -= ax;
      matrix.mmv(x, y);
      MatrixTestImpl::expectNear(y, expected, tol, "mmv() produced incorrect result");
    }

    // y += alpha A x
    {
      const typename Matrix::field_type alpha(0.75);
      RangeVector y = y0;
      RangeVector expected = y0;
      RangeVector scaled = ax;
      scaled *= alpha;
      expected += scaled;
      matrix.usmv(alpha, x, y);
      MatrixTestImpl::expectNear(y, expected, tol, "usmv() produced incorrect result");
    }

    // y += A^T z
    {
      DomainVector y = yt0;
      DomainVector expected = yt0;
      expected += atz;
      matrix.umtv(z, y);
      MatrixTestImpl::expectNear(y, expected, tol, "umtv() produced incorrect result");
    }

    // y -= A^T z
    {
      DomainVector y = yt0;
      DomainVector expected = yt0;
      expected -= atz;
      matrix.mmtv(z, y);
      MatrixTestImpl::expectNear(y, expected, tol, "mmtv() produced incorrect result");
    }

    // y += alpha A^T z
    {
      const typename Matrix::field_type alpha(0.75);
      DomainVector y = yt0;
      DomainVector expected = yt0;
      DomainVector scaled = atz;
      scaled *= alpha;
      expected += scaled;
      matrix.usmtv(alpha, z, y);
      MatrixTestImpl::expectNear(y, expected, tol, "usmtv() produced incorrect result");
    }

    // y += A^H z
    {
      DomainVector y = yt0;
      DomainVector expected = yt0;
      expected += ahz;
      matrix.umhv(z, y);
      MatrixTestImpl::expectNear(y, expected, tol, "umhv() produced incorrect result");
    }

    // y -= A^H z
    {
      DomainVector y = yt0;
      DomainVector expected = yt0;
      expected -= ahz;
      matrix.mmhv(z, y);
      MatrixTestImpl::expectNear(y, expected, tol, "mmhv() produced incorrect result");
    }

    // y += alpha A^H z
    {
      const typename Matrix::field_type alpha(0.75);
      DomainVector y = yt0;
      DomainVector expected = yt0;
      DomainVector scaled = ahz;
      scaled *= alpha;
      expected += scaled;
      matrix.usmhv(alpha, z, y);
      MatrixTestImpl::expectNear(y, expected, tol, "usmhv() produced incorrect result");
    }

    // Inner-product consistency checks on blocked vectors.
    {
      auto lhs = ax.dot(z);
      auto rhsHermitian = x.dot(ahz);
      using std::max;
      using std::abs;
      const auto hermitianResidual = abs(lhs - rhsHermitian) / max(real_type{1.}, abs(rhsHermitian));
      if (hermitianResidual > 1e-9)
        DUNE_THROW(ISTLError, "Hermitian consistency check failed");

      if constexpr (std::is_same<field_type, real_type>::value)
      {
        const auto rhsTranspose = x.dot(atz);
        const auto transposeResidual = abs(lhs - rhsTranspose) / max(real_type{1.}, abs(rhsTranspose));
        if (transposeResidual > 1e-9)
          DUNE_THROW(ISTLError, "Transpose consistency check failed");
      }
    }
  }

  /** \brief Test whether a given type implements all the norms required from a dune-istl matrix
   */
  template <typename Matrix>
  void testNorms(const Matrix& m)
  {
    using field_type = typename Matrix::field_type;
    using real_type = typename FieldTraits<field_type>::real_type;

    // frobenius_norm
    static_assert(std::is_same<decltype(m.frobenius_norm()),real_type>::value, "'frobenius_norm' does not return 'real_type'");
    if (m.frobenius_norm() < 0.0)
      DUNE_THROW(RangeError, "'frobenius_norm' returns negative value");

    // frobenius_norm2
    static_assert(std::is_same<decltype(m.frobenius_norm2()),real_type>::value, "'frobenius_norm2' does not return 'real_type'");
    if (m.frobenius_norm2() < 0.0)
      DUNE_THROW(RangeError, "'frobenius_norm2' returns negative value");

    // infinity_norm
    static_assert(std::is_same<decltype(m.infinity_norm()),real_type>::value, "'infinity_norm' does not return 'real_type'");
    if (m.infinity_norm() < 0.0)
      DUNE_THROW(RangeError, "'infinity_norm' returns negative value");

    // infinity_norm_real
    static_assert(std::is_same<decltype(m.infinity_norm_real()),real_type>::value, "'infinity_norm_real' does not return 'real_type'");
    if (m.infinity_norm_real() < 0.0)
      DUNE_THROW(RangeError, "'infinity_norm_real' returns negative value");
  }

  template <typename Matrix>
  void testMatrixTranspose(const Matrix& matrix,
                           typename FieldTraits<typename Matrix::field_type>::real_type tol = 1e-10)
  {
    Matrix transposedMatrix = matrix.transpose();

    for (std::size_t i = 0; i < matrix.N(); ++i)
      for (std::size_t j = 0; j < matrix.M(); ++j) {
        auto diff = transposedMatrix[j][i];
        diff -= matrix[i][j];
        using std::max;
        const auto err = Impl::asMatrix(diff).frobenius_norm();
        const auto scale = max(decltype(err)(1), Impl::asMatrix(matrix[i][j]).frobenius_norm());
        if (err > tol * scale)
          DUNE_THROW(ISTLError, "transpose() method produces wrong result!");
      }
  }

  template <typename Matrix, typename Vector>
  void testMatrixSolve(const Matrix& matrix,
                       typename FieldTraits<typename Matrix::field_type>::real_type tol = 1e-10)
  {
    using size_type = typename Vector::size_type;

    Vector b(matrix.N());
    for (size_type i = 0; i < b.size(); ++i)
      b[i] = i;

    Vector x(matrix.M());
    matrix.solve(x, b);

    matrix.mmv(x, b);

    if (b.two_norm() > tol)
      DUNE_THROW(ISTLError, "Solve() method doesn't appear to produce the solution!");
  }

  template <typename Matrix>
  void testVectorSpaceOperations(const Matrix& m)
  {
    using real_type = typename FieldTraits<typename Matrix::field_type>::real_type;
    const real_type tol = 1e-11;

    // Check that two matrices (same sparsity) are close in the Frobenius norm.
    auto checkNear = [&](const Matrix& computed, const Matrix& expected, const char* msg) {
      Matrix diff = computed;
      diff -= expected;
      const real_type scale = std::max(real_type(1), expected.frobenius_norm());
      const real_type relErr = diff.frobenius_norm() / scale;
      if (relErr > tol)
        DUNE_THROW(ISTLError, msg << " (relative error " << relErr << ")");
    };

    // operator+=: m + m  ==  2 * m
    {
      Matrix result = m;
      result += m;
      Matrix expected = m;
      expected *= 2.0;
      checkNear(result, expected, "operator+= produced incorrect result");
    }

    // operator-=: m - m  ==  0
    {
      Matrix result = m;
      result -= m;
      const real_type scale = std::max(real_type(1), m.frobenius_norm());
      if (result.frobenius_norm() > tol * scale)
        DUNE_THROW(ISTLError, "operator-= produced incorrect result (m - m != 0)");
    }

    // operator*=: (m * 0.5) * 2  ==  m
    {
      Matrix result = m;
      result *= 0.5;
      result *= 2.0;
      checkNear(result, m, "operator*= produced incorrect result");
    }

    // operator/=: (m / 2) * 2  ==  m
    {
      Matrix result = m;
      result /= 2.0;
      result *= 2.0;
      checkNear(result, m, "operator/= produced incorrect result");
    }

    // axpy: result.axpy(2, m) starting from result == m  =>  result == 3*m
    if constexpr (requires (Matrix a, const Matrix& b) { a.axpy(1.0, b); })
    {
      Matrix result = m;
      result.axpy(2.0, m);
      Matrix expected = m;
      expected *= 3.0;
      checkNear(result, expected, "axpy() produced incorrect result");
    }
  }

  /** \brief Test a matrix (view) against its explicitly supplied domain and range vectors.
   *
   *  Checks type exports, forward and backward iterators, assignment/copy,
   *  and delegates to the shared interface helpers.
   */
  template <class MatrixType, class X, class Y>
  void testMatrixView(MatrixType& matrix, X& x, Y& y)
  {
    using block_type [[maybe_unused]] = typename MatrixType::block_type;
    using row_type [[maybe_unused]] = typename MatrixType::row_type;
    using RowIterator [[maybe_unused]] = typename MatrixType::RowIterator;
    using ConstRowIterator [[maybe_unused]] = typename MatrixType::ConstRowIterator;
    using ColIterator [[maybe_unused]] = typename MatrixType::ColIterator;
    using ConstColIterator [[maybe_unused]] = typename MatrixType::ConstColIterator;

    static_assert(maxBlockLevel<MatrixType>() >= 0, "Block level has to be at least 1 for a matrix!");

    // ////////////////////////////////////////////////////////
    //   Count rows and entries using forward iterators
    // ////////////////////////////////////////////////////////

    typename MatrixType::RowIterator rowIt    = matrix.begin();
    typename MatrixType::RowIterator rowEndIt = matrix.end();
    typename MatrixType::size_type numRows = 0, numEntries = 0;

    for (; rowIt != rowEndIt; ++rowIt) {
      typename MatrixType::ColIterator colIt    = rowIt->begin();
      typename MatrixType::ColIterator colEndIt = rowIt->end();
      for (; colIt != colEndIt; ++colIt) {
        assert(matrix.exists(rowIt.index(), colIt.index()));
        numEntries++;
      }
      numRows++;
    }
    assert(numRows == matrix.N());

    // ////////////////////////////////////////////////////////
    //   Same with const iterators
    // ////////////////////////////////////////////////////////

    typename MatrixType::ConstRowIterator constRowIt    = matrix.begin();
    typename MatrixType::ConstRowIterator constRowEndIt = matrix.end();
    numRows = 0; numEntries = 0;

    for (; constRowIt != constRowEndIt; ++constRowIt) {
      typename MatrixType::ConstColIterator constColIt    = constRowIt->begin();
      typename MatrixType::ConstColIterator constColEndIt = constRowIt->end();
      for (; constColIt != constColEndIt; ++constColIt)
        numEntries++;
      numRows++;
    }
    assert(numRows == matrix.N());

    // ////////////////////////////////////////////////////////
    //   Backward iteration
    // ////////////////////////////////////////////////////////

    if constexpr (requires { matrix.beforeBegin(); matrix.beforeEnd(); }) {
      rowIt    = matrix.beforeEnd();
      rowEndIt = matrix.beforeBegin();
      numRows = 0; numEntries = 0;

      for (; rowIt != rowEndIt; --rowIt) {
        typename MatrixType::ColIterator colIt    = rowIt->beforeEnd();
        typename MatrixType::ColIterator colEndIt = rowIt->beforeBegin();
        for (; colIt != colEndIt; --colIt) {
          assert(matrix.exists(rowIt.index(), colIt.index()));
          numEntries++;
        }
        numRows++;
      }
      assert(numRows == matrix.N());

      constRowIt    = matrix.beforeEnd();
      constRowEndIt = matrix.beforeBegin();
      numRows = 0; numEntries = 0;

      for (; constRowIt != constRowEndIt; --constRowIt) {
        typename MatrixType::ConstColIterator constColIt    = constRowIt->beforeEnd();
        typename MatrixType::ConstColIterator constColEndIt = constRowIt->beforeBegin();
        for (; constColIt != constColEndIt; --constColIt)
          numEntries++;
        numRows++;
      }
      assert(numRows == matrix.N());
    }

    // Shared interface tests.
    testVectorSpaceOperations(matrix);
    testNorms(matrix);
    testMatrixVectorProducts(matrix, x, y);

    // Scalar assignment
    using real_type = typename FieldTraits<typename MatrixType::field_type>::real_type;

    matrix = real_type{0};
    if (matrix.frobenius_norm() != real_type{0})
      DUNE_THROW(ISTLError, "Scalar assignment to zero failed");
  }


  template <class MatrixType, class X, class Y>
  void testMatrix(MatrixType& matrix, X& x, Y& y) {

    testMatrixView(matrix, x, y);

    // ///////////////////////////////////////////////////////
    //   Test assignment operators and the copy constructor
    // ///////////////////////////////////////////////////////

    testMatrixConstructibility<MatrixType>();

    MatrixType secondMatrix;
    secondMatrix = matrix;
    testMatrixView(secondMatrix, x, y);

    MatrixType thirdMatrix(matrix);
    testMatrixView(thirdMatrix, x, y);
  }

  /** \brief Test a matrix with blocked entries, constructing suitable vectors automatically.
   *
   *  Requires that `block_type` exposes static `rows` and `cols` members,
   *  i.e., the block type is a fixed-size matrix such as `FieldMatrix<K,r,c>`.
   */
  template <class MatrixType>
  void testSuperMatrix(MatrixType& matrix)
  {
    using field_type = typename MatrixType::field_type;
    using block_type = typename MatrixType::block_type;

    typename MatrixType::size_type n = matrix.N();
    typename MatrixType::size_type m = matrix.M();
    BlockVector<FieldVector<field_type, block_type::cols> > x(m);
    BlockVector<FieldVector<field_type, block_type::rows> > y(n);

    testMatrix(matrix, x, y);
  }

}  // namespace Dune

#endif
