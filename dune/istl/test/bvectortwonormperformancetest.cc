// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>

#include <dune/common/alignedallocator.hh>
#include <dune/common/float_cmp.hh>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/timer.hh>

#include <dune/istl/bvector.hh>

volatile void* doNotOptimizeAway = nullptr;

template <class V>
decltype(std::declval<const V&>().two_norm()) naiveTwoNorm(const V& x)
{
  using Real = decltype(x.two_norm());
  Real sum(0);

  for (typename V::size_type i = 0; i < x.N(); ++i)
    sum += x[i].two_norm2();

  return std::sqrt(sum);
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
double benchmarkVectorTwoNorm(const V& x, std::size_t iterations)
{
  for (std::size_t i = 0; i < 10; ++i) {
    auto warmup = x.two_norm();
    doNotOptimizeAway = &warmup;
  }

  Dune::MPIHelper::getCommunication().barrier();
  Dune::Timer timer;
  auto sink = decltype(x.two_norm())(0);
  for (std::size_t it = 0; it < iterations; ++it)
    sink += x.two_norm();

  doNotOptimizeAway = &sink;

  auto result = timer.elapsed() / static_cast<double>(iterations);
  Dune::MPIHelper::getCommunication().barrier();
  return result;
}

template <class V>
double benchmarkNaiveTwoNorm(const V& x, std::size_t iterations)
{
  for (std::size_t i = 0; i < 10; ++i) {
    auto warmup = naiveTwoNorm(x);
    doNotOptimizeAway = &warmup;
  }

  Dune::MPIHelper::getCommunication().barrier();
  Dune::Timer timer;
  auto sink = decltype(x.two_norm())(0);
  for (std::size_t it = 0; it < iterations; ++it)
    sink += naiveTwoNorm(x);

  doNotOptimizeAway = &sink;

  auto result = timer.elapsed() / static_cast<double>(iterations);
  Dune::MPIHelper::getCommunication().barrier();
  return result;
}

template <int blockSize>
Dune::TestSuite runCase(std::size_t nBlocks, std::size_t iterations)
{
  Dune::TestSuite t("BlockVector two_norm test (block size " + std::to_string(blockSize) + ")");

  using Block = Dune::FieldVector<double, blockSize>;
  using Vector = Dune::BlockVector<Block>;

  Vector x(nBlocks);
  fillRandom(x, 9000 + blockSize);

  const double naive = naiveTwoNorm(x);
  const double optimized = x.two_norm();

  t.check(Dune::FloatCmp::eq(optimized, naive, 1e-10))
      << "two_norm mismatch for block size " << blockSize
      << ": optimized=" << optimized << ", naive=" << naive;

  const double tOptimized = benchmarkVectorTwoNorm(x, iterations);
  const double tNaive = benchmarkNaiveTwoNorm(x, iterations);
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
  Dune::TestSuite t("BlockVector two_norm aligned window test (block size " + std::to_string(blockSize) + ")");

  using Block = Dune::FieldVector<double, blockSize>;
  using Window = Dune::Imp::BlockVectorWindow<Block, Dune::AlignedAllocator<Block, 64>>;

  Dune::BlockVector<Block> storage(nBlocks);
  fillRandom(storage, 9000 + blockSize);

  constexpr std::size_t alignment = 64;

  std::size_t space = sizeof(Block) * nBlocks;
  void* ptr = static_cast<void*>(storage.data());
  ptr = std::align(alignment, sizeof(Block), ptr, space);

  if (Dune::MPIHelper::getCommunication().min(static_cast<int>(ptr != nullptr)) == 0)
    return t;

  constexpr std::size_t blockBytes = sizeof(Block);
  const std::size_t blocksPerAlignedSpan = alignment / std::gcd(alignment, blockBytes);

  const std::size_t blocks = space / blockBytes;
  std::size_t windowBlocks = blocks - blocks % blocksPerAlignedSpan;

  if (Dune::MPIHelper::getCommunication().min(static_cast<int>(windowBlocks != 0)) == 0)
    return t;

  Window x(static_cast<Block*>(ptr), windowBlocks);

  const double naive = naiveTwoNorm(x);
  const double optimized = x.two_norm();

  t.check(Dune::FloatCmp::eq(optimized, naive, 1e-10))
      << "aligned window two_norm mismatch for block size " << blockSize
      << ": optimized=" << optimized << ", naive=" << naive;

  const double tOptimized = benchmarkVectorTwoNorm(x, iterations);
  const double tNaive = benchmarkNaiveTwoNorm(x, iterations);
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
  Dune::TestSuite t("BlockVector two_norm unaligned window test (block size " + std::to_string(blockSize) + ")");

  using Block = Dune::FieldVector<double, blockSize>;
  using Window = Dune::Imp::BlockVectorWindow<Block, Dune::AlignedAllocator<Block, 64>>;

  Dune::BlockVector<Block> storage(nBlocks);
  fillRandom(storage, 9000 + blockSize);

  constexpr std::size_t alignment = 64;

  std::size_t space = sizeof(Block) * nBlocks;
  void* ptr = static_cast<void*>(storage.data());
  ptr = std::align(alignment, sizeof(Block), ptr, space);

  if (Dune::MPIHelper::getCommunication().min(static_cast<int>(ptr != nullptr)) == 0)
    return t;

  constexpr std::size_t blockBytes = sizeof(Block);
  const std::size_t blocksPerAlignedSpan = alignment / std::gcd(alignment, blockBytes);

  const std::size_t blocks = space / blockBytes;
  std::size_t windowBlocks = blocks - blocks % blocksPerAlignedSpan;

  if (Dune::MPIHelper::getCommunication().min(static_cast<int>(windowBlocks > 2)) == 0)
    return t;

  ptr = static_cast<void*>(static_cast<Block*>(ptr) + 1);
  windowBlocks -= 2;

  Window x(static_cast<Block*>(ptr), windowBlocks);

  const double naive = naiveTwoNorm(x);
  const double optimized = x.two_norm();

  t.check(Dune::FloatCmp::eq(optimized, naive, 1e-10))
      << "unaligned window two_norm mismatch for block size " << blockSize
      << ": optimized=" << optimized << ", naive=" << naive;

  const double tOptimized = benchmarkVectorTwoNorm(x, iterations);
  const double tNaive = benchmarkNaiveTwoNorm(x, iterations);
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
  for (std::size_t b = 100; b <= 1'000'000; b *= 10)
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
