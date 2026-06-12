// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

#include <dune/common/float_cmp.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/timer.hh>
#include <dune/common/fvector.hh>
#include <dune/common/alignedallocator.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/istl/bvector.hh>

volatile void* doNotOptimizeAway = nullptr;

template <class V>
typename V::field_type naiveDot(const V& x, const V& y)
{
  using T = typename V::field_type;
  T sum(0);

  for (typename V::size_type i = 0; i < x.N(); ++i)
    sum += x[i].dot(y[i]);

  return sum;
}

template <class V>
void fillRandom(V& v, unsigned seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<typename V::field_type> dist(-1.0, 1.0);

  for (auto& block : v) {
    for (auto& value : block)
      value = dist(gen);
  }
}

template <class V>
double benchmarkVectorDot(const V& x, const V& y, std::size_t iterations)
{
  for (std::size_t i = 0; i < 10; ++i) {
    auto warmup = x.dot(y);
    doNotOptimizeAway = &warmup;
  }

  Dune::MPIHelper::getCommunication().barrier();
  Dune::Timer timer;
  typename V::field_type sink(0);
  for (std::size_t it = 0; it < iterations; ++it)
    sink += x.dot(y);

  doNotOptimizeAway = &sink;

  auto result = timer.elapsed() / static_cast<double>(iterations);
  Dune::MPIHelper::getCommunication().barrier();
  return result;
}

template <class V>
double benchmarkNaiveDot(const V& x, const V& y, std::size_t iterations)
{
  for (std::size_t i = 0; i < 10; ++i) {
    auto warmup = naiveDot(x, y);
    doNotOptimizeAway = &warmup;
  }

  Dune::MPIHelper::getCommunication().barrier();
  Dune::Timer timer;
  typename V::field_type sink(0);
  for (std::size_t it = 0; it < iterations; ++it)
    sink += naiveDot(x, y);

  doNotOptimizeAway = &sink;

  auto result = timer.elapsed() / static_cast<double>(iterations);
  Dune::MPIHelper::getCommunication().barrier();
  return result;
}

template <int blockSize>
Dune::TestSuite runCase(std::size_t nBlocks, std::size_t iterations)
{
  Dune::TestSuite t("BlockVector dot test (block size " + std::to_string(blockSize) + ")");

  using Block = Dune::FieldVector<double, blockSize>;
  using Vector = Dune::BlockVector<Block>;

  Vector x(nBlocks), y(nBlocks);
  fillRandom(x, 1234 + blockSize);
  fillRandom(y, 5678 + blockSize);

  const double naive = naiveDot(x, y);
  const double optimized = x.dot(y);

  t.check(Dune::FloatCmp::eq(optimized, naive, 1e-10))
      << "dot mismatch for block size " << blockSize
      << ": optimized=" << optimized << ", naive=" << naive;

  const double tOptimized = benchmarkVectorDot(x, y, iterations);
  const double tNaive = benchmarkNaiveDot(x, y, iterations);
  const double speedup = tNaive / tOptimized;

  std::cout << std::fixed << std::setprecision(6)
            << "BlockSize=" << blockSize
            << " nBlocks=" << nBlocks
            << " optimized=" << tOptimized * 1e3 << " ms"
            << " naive=" << tNaive * 1e3 << " ms"
            << " speedup=" << speedup << "x"
            << std::endl;

  return t;
}

template <int blockSize>
Dune::TestSuite runAlignedWindowCase(std::size_t nBlocks, std::size_t iterations)
{
  Dune::TestSuite t("BlockVector dot aligned window test (block size " + std::to_string(blockSize) + ")");

  using Block = Dune::FieldVector<double, blockSize>;
  using Window = Dune::Imp::BlockVectorWindow<Block, Dune::AlignedAllocator<Block, 64>>;

  // Create extra head/tail blocks and take an interior window.
  // This intentionally forces begin/end alignment assumptions in optimized loops.

  Dune::BlockVector<Block> xStorage(nBlocks), yStorage(nBlocks);

  fillRandom(xStorage, 9000 + blockSize);
  fillRandom(yStorage, 10000 + blockSize);

  constexpr std::size_t alignment = 64;

  std::size_t xSpace = sizeof(Block) * nBlocks;
  void* xPtr = (void*)xStorage.data();
  xPtr = std::align(alignment, sizeof(Block), xPtr, xSpace);
  std::size_t ySpace = sizeof(Block) * nBlocks;
  void* yPtr = (void*)yStorage.data();
  yPtr = std::align(alignment, sizeof(Block), yPtr, ySpace);


  if (Dune::MPIHelper::getCommunication().min((int)(xPtr && yPtr)) == 0)
    return t;

  constexpr std::size_t blockBytes = sizeof(Block);
  const std::size_t blocksPerAlignedSpan = alignment / std::gcd(alignment, blockBytes);

  const std::size_t xBlocks = xSpace / blockBytes;
  const std::size_t yBlocks = ySpace / blockBytes;
  std::size_t windowBlocks = std::min(xBlocks, yBlocks);
  windowBlocks -= windowBlocks % blocksPerAlignedSpan;

  if (Dune::MPIHelper::getCommunication().min((int)(windowBlocks != 0)) == 0)
    return t;

  Window x((Block*)xPtr, windowBlocks);
  Window y((Block*)yPtr, windowBlocks);

  const double naive = naiveDot(x, y);
  const double optimized = x.dot(y);

  t.check(Dune::FloatCmp::eq(optimized, naive, 1e-10))
      << "aligned window dot mismatch for block size " << blockSize
      << ": optimized=" << optimized << ", naive=" << naive;

  const double tOptimized = benchmarkVectorDot(x, y, iterations);
  const double tNaive = benchmarkNaiveDot(x, y, iterations);
  const double speedup = tNaive / tOptimized;

  std::cout << std::fixed << std::setprecision(6)
            << "AlignedWindow BlockSize=" << blockSize
            << " nBlocks=" << windowBlocks
            << " optimized=" << tOptimized * 1e3 << " ms"
            << " naive=" << tNaive * 1e3 << " ms"
            << " speedup=" << speedup << "x"
            << std::endl;

  return t;
}

template <int blockSize>
Dune::TestSuite runUnalignedWindowCase(std::size_t nBlocks, std::size_t iterations)
{
  Dune::TestSuite t("BlockVector dot unaligned window test (block size " + std::to_string(blockSize) + ")");

  using Block = Dune::FieldVector<double, blockSize>;
  using Window = Dune::Imp::BlockVectorWindow<Block, Dune::AlignedAllocator<Block, 64>>;

  // Create extra head/tail blocks and take an interior window.
  // This intentionally forces begin/end alignment assumptions in optimized loops.

  Dune::BlockVector<Block> xStorage(nBlocks), yStorage(nBlocks);

  fillRandom(xStorage, 9000 + blockSize);
  fillRandom(yStorage, 10000 + blockSize);

  constexpr std::size_t alignment = 64;

  std::size_t xSpace = sizeof(Block) * nBlocks;
  void* xPtr = (void*)xStorage.data();
  xPtr = std::align(alignment, sizeof(Block), xPtr, xSpace);
  std::size_t ySpace = sizeof(Block) * nBlocks;
  void* yPtr = (void*)yStorage.data();
  yPtr = std::align(alignment, sizeof(Block), yPtr, ySpace);


  if (Dune::MPIHelper::getCommunication().min((int)(xPtr && yPtr)) == 0)
    return t;

  constexpr std::size_t blockBytes = sizeof(Block);
  const std::size_t blocksPerAlignedSpan = alignment / std::gcd(alignment, blockBytes);

  const std::size_t xBlocks = xSpace / blockBytes;
  const std::size_t yBlocks = ySpace / blockBytes;
  std::size_t windowBlocks = std::min(xBlocks, yBlocks);
  windowBlocks -= windowBlocks % blocksPerAlignedSpan;

  if (Dune::MPIHelper::getCommunication().min((int)(windowBlocks > 2)) == 0)
    return t;

  xPtr = (void*)((Block*)xPtr + 1); // make unaligned by skipping one block from the start of the window
  yPtr = (void*)((Block*)yPtr + 1); // make unaligned by skipping one block from the start of the window
  windowBlocks -= 2; // make unaligned by removing one block from the window (plus the one we skipped from the start)

  Window x(((Block*)xPtr), windowBlocks);
  Window y(((Block*)yPtr), windowBlocks);

  const double naive = naiveDot(x, y);
  const double optimized = x.dot(y);

  t.check(Dune::FloatCmp::eq(optimized, naive, 1e-10))
      << "unaligned window dot mismatch for block size " << blockSize
      << ": optimized=" << optimized << ", naive=" << naive;

  const double tOptimized = benchmarkVectorDot(x, y, iterations);
  const double tNaive = benchmarkNaiveDot(x, y, iterations);
  const double speedup = tNaive / tOptimized;

  std::cout << std::fixed << std::setprecision(6)
            << "UnalignedWindow BlockSize=" << blockSize
            << " nBlocks=" << windowBlocks
            << " optimized=" << tOptimized * 1e3 << " ms"
            << " naive=" << tNaive * 1e3 << " ms"
            << " speedup=" << speedup << "x"
            << std::endl;

  return t;
}

int main(int argc, char** argv)
{
  auto& mpiHelper = Dune::MPIHelper::instance(argc, argv);

  Dune::TestSuite t;

#ifdef NDEBUG
  constexpr std::size_t iterations = 50;
  std::vector<std::size_t> blockSizes = {1, 2, 4, 8, 16, 32};
  for (std::size_t b = 100 ; b <= 1'000'000; b *= 10)
    blockSizes.push_back(b);
#else
  constexpr std::size_t iterations = 15;
  std::vector<std::size_t> blockSizes = {1, 2, 4, 8, 16, 32};
#endif
  for (auto nBlocks : blockSizes) {
    if (nBlocks == 0)
      nBlocks = 1;
    t.subTest(runCase<1>(nBlocks, iterations));
    t.subTest(runCase<2>(nBlocks, iterations));
    t.subTest(runCase<3>(nBlocks, iterations));
    t.subTest(runCase<4>(nBlocks, iterations));
    t.subTest(runCase<5>(nBlocks, iterations));
    t.subTest(runCase<6>(nBlocks, iterations));
    t.subTest(runCase<7>(nBlocks, iterations));
    t.subTest(runCase<8>(nBlocks, iterations));

    t.subTest(runAlignedWindowCase<1>(nBlocks, iterations));
    t.subTest(runAlignedWindowCase<2>(nBlocks, iterations));
    t.subTest(runAlignedWindowCase<3>(nBlocks, iterations));
    t.subTest(runAlignedWindowCase<4>(nBlocks, iterations));
    t.subTest(runAlignedWindowCase<5>(nBlocks, iterations));
    t.subTest(runAlignedWindowCase<6>(nBlocks, iterations));
    t.subTest(runAlignedWindowCase<7>(nBlocks, iterations));
    t.subTest(runAlignedWindowCase<8>(nBlocks, iterations));

    t.subTest(runUnalignedWindowCase<1>(nBlocks, iterations));
    t.subTest(runUnalignedWindowCase<2>(nBlocks, iterations));
    t.subTest(runUnalignedWindowCase<3>(nBlocks, iterations));
    t.subTest(runUnalignedWindowCase<4>(nBlocks, iterations));
    t.subTest(runUnalignedWindowCase<5>(nBlocks, iterations));
    t.subTest(runUnalignedWindowCase<6>(nBlocks, iterations));
    t.subTest(runUnalignedWindowCase<7>(nBlocks, iterations));
    t.subTest(runUnalignedWindowCase<8>(nBlocks, iterations));
  }

  return t.exit();
}
