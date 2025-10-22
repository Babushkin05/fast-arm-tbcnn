#include <catch2/catch_all.hpp>
#include "GeMM.hpp"

TEST_CASE("Naive GeMM basic 2x2") {
    int32_t A[] = {1, 2, 3, 4};
    int32_t B[] = {5, 6, 7, 8};
    auto C = GeMM(A, B, 2, 2, 2);

    REQUIRE(C[0] == 19);
    REQUIRE(C[1] == 22);
    REQUIRE(C[2] == 43);
    REQUIRE(C[3] == 50);

    delete[] C;
}
