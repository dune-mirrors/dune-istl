// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/float_cmp.hh>

#include <array>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/test/laplacian.hh>
#include <dune/istl/test/matrixtest.hh>

using namespace Dune;

template <class Matrix, class Vector>
int testBCRSMatrix(int size)
{
  // Set up a test matrix
  Matrix mat;
  setupLaplacian(mat, size);

  // Test vector space operations
  testVectorSpaceOperations(mat);

  // Test the matrix norms
  testNorms(mat);

  // Test whether matrix class has the required constructors
  testMatrixConstructibility<Matrix>();

  // Test the matrix vector products
  Vector domain(mat.M());
  domain = 0;
  Vector range(mat.N());

  testMatrixVectorProducts(mat,domain,range);

  return 0;
}

int main(int argc, char** argv)
{
  // Test scalar matrices and vectors
  int ret = testBCRSMatrix<BCRSMatrix<double>, BlockVector<double> >(10);

  // Test block matrices and vectors with trivial blocks
  ret = testBCRSMatrix<BCRSMatrix<FieldMatrix<double,1,1> >, BlockVector<FieldVector<double,1> > >(10);

  // Test scalar matrices with complex numbers
  ret = testBCRSMatrix<BCRSMatrix<std::complex<double>>, BlockVector<std::complex<double>>>(10);

  // Dimension-sensitive check: non-square matrix catches swapped domain/range arguments.
  {
    using Matrix = BCRSMatrix<double>;
    constexpr std::size_t rows = 4;
    constexpr std::size_t cols = 7;

    Matrix mat;
    mat.setBuildMode(Matrix::row_wise);
    mat.setSize(rows, cols, rows * 3);

    for (auto row = mat.createbegin(); row != mat.createend(); ++row)
    {
      const std::size_t i = row.index();
      const std::array<std::size_t, 3> pattern = {i % cols, (i + 2) % cols, (i + 4) % cols};
      for (const auto j : pattern)
        row.insert(j);
    }

    for (auto row = mat.begin(); row != mat.end(); ++row)
      for (auto col = row->begin(); col != row->end(); ++col)
        *col = 0.2 * static_cast<double>(1 + row.index() + 2 * col.index());

    BlockVector<double> domain(cols);
    BlockVector<double> range(rows);
    testMatrixVectorProducts(mat, domain, range);
  }

  // ////////////////////////////////////////////////////////////
  //   Test the BCRSMatrix class -- a sparse dynamic matrix
  // ////////////////////////////////////////////////////////////

  {
    BCRSMatrix<double> bcrsMatrix(4,4, BCRSMatrix<double>::random);

    bcrsMatrix.setrowsize(0,2);
    bcrsMatrix.setrowsize(1,3);
    bcrsMatrix.setrowsize(2,3);
    bcrsMatrix.setrowsize(3,2);

    bcrsMatrix.endrowsizes();

    bcrsMatrix.addindex(0, 0);
    bcrsMatrix.addindex(0, 1);

    bcrsMatrix.addindex(1, 0);
    bcrsMatrix.addindex(1, 1);
    bcrsMatrix.addindex(1, 2);

    bcrsMatrix.addindex(2, 1);
    bcrsMatrix.addindex(2, 2);
    bcrsMatrix.addindex(2, 3);

    bcrsMatrix.addindex(3, 2);
    bcrsMatrix.addindex(3, 3);

    bcrsMatrix.endindices();

    for (auto row = bcrsMatrix.begin(); row != bcrsMatrix.end(); ++row)
      for (auto col = row->begin(); col != row->end(); ++col)
        *col = 1.0 + static_cast<double>(row.index()) * static_cast<double>(col.index());

    BlockVector<double> x(4), y(4);
    testMatrix(bcrsMatrix, x, y);

    // Test whether matrix resizing works
    int resize = 3;
    bcrsMatrix.setSize(resize, resize, resize);

    for (int i = 0; i < resize; i++)
      bcrsMatrix.setrowsize(i, 1);
    bcrsMatrix.endrowsizes();

    for (int i = 0; i < resize; i++)
      bcrsMatrix.addindex(i, i);
    bcrsMatrix.endindices();

    for (int i = 0; i < resize; i++)
      bcrsMatrix[i][i] = 1.0;

    x.resize(resize);
    y.resize(resize);
    testMatrix(bcrsMatrix, x, y);
  }

  // ////////////////////////////////////////////////////////////
  //   Test the BCRSMatrix class with FieldMatrix entries
  // ////////////////////////////////////////////////////////////

  {
    BCRSMatrix<FieldMatrix<double,2,2>> bcrsMatrix2x2(4, 4, BCRSMatrix<FieldMatrix<double,2,2>>::random);

    bcrsMatrix2x2.setrowsize(0,2);
    bcrsMatrix2x2.setrowsize(1,3);
    bcrsMatrix2x2.setrowsize(2,3);
    bcrsMatrix2x2.setrowsize(3,2);

    bcrsMatrix2x2.endrowsizes();

    bcrsMatrix2x2.addindex(0, 0);
    bcrsMatrix2x2.addindex(0, 1);

    bcrsMatrix2x2.addindex(1, 0);
    bcrsMatrix2x2.addindex(1, 1);
    bcrsMatrix2x2.addindex(1, 2);

    bcrsMatrix2x2.addindex(2, 1);
    bcrsMatrix2x2.addindex(2, 2);
    bcrsMatrix2x2.addindex(2, 3);

    bcrsMatrix2x2.addindex(3, 2);
    bcrsMatrix2x2.addindex(3, 3);

    bcrsMatrix2x2.endindices();

    for (auto row = bcrsMatrix2x2.begin(); row != bcrsMatrix2x2.end(); ++row)
      for (auto col = row->begin(); col != row->end(); ++col)
        *col = 1.0 + static_cast<double>(row.index()) * static_cast<double>(col.index());

    testSuperMatrix(bcrsMatrix2x2);

    // Test whether matrix resizing works
    int resize = 3;
    bcrsMatrix2x2.setSize(resize, resize, resize);

    for (int i = 0; i < resize; i++)
      bcrsMatrix2x2.setrowsize(i, 1);
    bcrsMatrix2x2.endrowsizes();

    for (int i = 0; i < resize; i++)
      bcrsMatrix2x2.addindex(i, i);
    bcrsMatrix2x2.endindices();

    for (int i = 0; i < resize; i++)
      bcrsMatrix2x2[i][i] = 1.0;

    testSuperMatrix(bcrsMatrix2x2);
  }

  return ret;
}
