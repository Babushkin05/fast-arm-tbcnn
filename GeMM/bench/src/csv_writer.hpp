#pragma once

#include <cstdio>
#include <string>
#include <string_view>
#include <cstdint>

namespace bench {

/// Simple CSV writer for benchmark results
class CsvWriter {
public:
    explicit CsvWriter(const std::string& filename);
    ~CsvWriter();

    CsvWriter(const CsvWriter&) = delete;
    CsvWriter& operator=(const CsvWriter&) = delete;

    /// Write header row
    void write_header();

    /// Write a single result row
    void write_row(
        std::string_view device,
        std::string_view impl,
        std::string_view matrix_type,
        std::uint32_t m,
        std::uint32_t n,
        std::uint32_t k,
        std::uint32_t run,
        double time_ms,
        double gflops,
        std::uint32_t mblk,
        std::uint32_t nblk,
        std::uint32_t kblk,
        std::uint32_t mmk,
        std::uint32_t nmk
    );

private:
    FILE* file_;
};

} // namespace bench
