#include <catch2/catch_all.hpp>
#include "GeMM.hpp"
#include "matrices_to_test.hpp"

using namespace tbn;

// Helper to compare results with reference
void CompareWithReference(const Int32Matrix& result, std::span<const std::int8_t> expected) {
    REQUIRE(result.size() == expected.size());

    for (std::size_t idx = 0; idx < expected.size(); ++idx) {
        const std::uint32_t row = static_cast<std::uint32_t>(idx / result.cols());
        const std::uint32_t col = static_cast<std::uint32_t>(idx % result.cols());

        REQUIRE(result.sign_at(row, col) == expected[idx]);
    }
}

// ============================================================================
// GeMM tests
// ============================================================================

TEST_CASE("128x128 - single block") {
    constexpr std::uint32_t m = 128, n = 128, k = 128;

    auto A = TernaryMatrix::pack({::A, m * n}, m, n);
    auto B = BinaryMatrix::pack({::B, n * k}, n, k);

    GemmEngine engine;
    TilingParams params = {.mblk = 128, .nblk = 128, .kblk = 128, .mmk = 64, .nmk = 64};

    Int32Matrix result = engine.compute(A.view(), B.view(), params);
    CompareWithReference(result, {::C, m * k});
    CHECK(engine.last_flops() == m * n * k * 2);
}

TEST_CASE("128x128 - multiple blocks") {
    constexpr std::uint32_t m = 128, n = 128, k = 128;

    auto A = TernaryMatrix::pack({::A, m * n}, m, n);
    auto B = BinaryMatrix::pack({::B, n * k}, n, k);

    GemmEngine engine;
    Int32Matrix result = engine.compute(A.view(), B.view(), TilingParams::default_128x128());
    CompareWithReference(result, {::C, m * k});
}

TEST_CASE("128x128 - small microkernels") {
    constexpr std::uint32_t m = 128, n = 128, k = 128;

    auto A = TernaryMatrix::pack({::A, m * n}, m, n);
    auto B = BinaryMatrix::pack({::B, n * k}, n, k);

    GemmEngine engine;
    TilingParams params = {.mblk = 32, .nblk = 32, .kblk = 128, .mmk = 16, .nmk = 16};

    Int32Matrix result = engine.compute(A.view(), B.view(), params);
    CompareWithReference(result, {::C, m * k});
}

TEST_CASE("Default params") {
    constexpr std::uint32_t m = 128, n = 128, k = 128;

    auto A = TernaryMatrix::pack({::A, m * n}, m, n);
    auto B = BinaryMatrix::pack({::B, n * k}, n, k);

    GemmEngine engine;
    Int32Matrix result = engine.compute(A.view(), B.view());
    CompareWithReference(result, {::C, m * k});
}

// ============================================================================
// Exception tests
// ============================================================================

TEST_CASE("Exception on invalid ternary values") {
    std::array<std::int8_t, 8 * 64> invalid_data{};
    invalid_data[0] = 2;

    REQUIRE_THROWS_AS(
        TernaryMatrix::pack(invalid_data, 8, 64),
        std::invalid_argument
    );
}

TEST_CASE("Exception on invalid binary values") {
    std::array<std::int8_t, 64 * 8> invalid_data{};
    invalid_data[0] = 0;

    REQUIRE_THROWS_AS(
        BinaryMatrix::pack(invalid_data, 64, 8),
        std::invalid_argument
    );
}

TEST_CASE("Exception on invalid dimensions") {
    std::array<std::int8_t, 100> data{};

    REQUIRE_THROWS_AS(
        TernaryMatrix::pack(data, 10, 10),
        std::invalid_argument
    );
}

TEST_CASE("Exception on invalid tiling params") {
    constexpr std::uint32_t m = 128, n = 128, k = 128;

    auto A = TernaryMatrix::pack({::A, m * n}, m, n);
    auto B = BinaryMatrix::pack({::B, n * k}, n, k);

    GemmEngine engine;
    TilingParams bad_params = {.mblk = 64, .nblk = 64, .kblk = 64, .mmk = 32, .nmk = 32};

    REQUIRE_THROWS_AS(
        engine.compute(A.view(), B.view(), bad_params),
        std::invalid_argument
    );
}

// ============================================================================
// Matrix properties
// ============================================================================

TEST_CASE("Empty matrices") {
    TernaryMatrix empty_ternary;
    BinaryMatrix empty_binary;
    Int32Matrix empty_result;

    CHECK(empty_ternary.empty());
    CHECK(empty_binary.empty());
    CHECK(empty_result.empty());
}

TEST_CASE("Int32Matrix accessor") {
    Int32Matrix mat(4, 4);

    mat.at(0, 0) = 10;
    mat.at(1, 2) = -5;
    mat.at(3, 3) = 0;

    CHECK(mat.at(0, 0) == 10);
    CHECK(mat.at(1, 2) == -5);
    CHECK(mat.at(3, 3) == 0);

    CHECK(mat.sign_at(0, 0) == 1);
    CHECK(mat.sign_at(1, 2) == -1);
    CHECK(mat.sign_at(3, 3) == 0);
}

TEST_CASE("Move semantics") {
    constexpr std::uint32_t m = 128, n = 128, k = 128;

    auto A = TernaryMatrix::pack({::A, m * n}, m, n);
    auto B = BinaryMatrix::pack({::B, n * k}, n, k);

    GemmEngine engine;
    Int32Matrix result = engine.compute(A.view(), B.view());

    Int32Matrix moved = std::move(result);
    CompareWithReference(moved, {::C, m * k});
}
