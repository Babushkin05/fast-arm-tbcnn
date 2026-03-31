#include "csv_writer.hpp"
#include <stdexcept>

namespace bench {

CsvWriter::CsvWriter(const std::string& filename) {
    file_ = fopen(filename.c_str(), "w");
    if (!file_) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
}

CsvWriter::~CsvWriter() {
    if (file_) {
        fclose(file_);
    }
}

void CsvWriter::write_header() {
    fprintf(file_, "device,impl,matrix_type,m,n,k,run,time_ms,gflops,mblk,nblk,kblk,mmk,nmk\n");
}

void CsvWriter::write_row(
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
) {
    fprintf(file_, "%.*s,%.*s,%.*s,%u,%u,%u,%u,%.6f,%.6f,%u,%u,%u,%u,%u\n",
            static_cast<int>(device.size()), device.data(),
            static_cast<int>(impl.size()), impl.data(),
            static_cast<int>(matrix_type.size()), matrix_type.data(),
            m, n, k, run, time_ms, gflops,
            mblk, nblk, kblk, mmk, nmk);
    fflush(file_);
}

} // namespace bench
