#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <random>
#include <span>

#include "matrix_gen.hpp"
#include "csv_writer.hpp"
#include "runner.hpp"

// Include GeMM implementation based on compile-time flag
#if defined(USE_IMPL_02)
    #include "../../02-coded/GeMM.hpp"
    #define IMPL_NAME "02-coded"
#elif defined(USE_IMPL_03)
    #include "../../03-blocked/GeMM.hpp"
    #define IMPL_NAME "03-blocked"
#elif defined(USE_IMPL_04)
    #include "../../04-neon/GeMM.hpp"
    #define IMPL_NAME "04-neon"
#else
    #include "../../05-final/GeMM.hpp"
    #define IMPL_NAME "05-final"
#endif

namespace {

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --sizes <list>       Matrix sizes (comma-separated, m=n=k)\n"
              << "  --types <list>       Matrix types (comma-separated):\n"
              << "                         random_dense, random_sparse, dense_no_zero,\n"
              << "                         diagonal, banded, block_sparse\n"
              << "  --runs <n>           Number of runs per size/type (default: 100)\n"
              << "  --warmup <n>         Warmup runs (default: 5)\n"
              << "  --pause-ms <ms>      Pause between runs (default: 50)\n"
              << "  --device <name>      Device identifier (default: unknown)\n"
              << "  --output <file>      Output CSV file (default: results.csv)\n"
              << "  --mblk <n>           Tiling: mblk (default: 64)\n"
              << "  --nblk <n>           Tiling: nblk (default: 64)\n"
              << "  --kblk <n>           Tiling: kblk (default: 128)\n"
              << "  --mmk <n>            Tiling: mmk (default: 32)\n"
              << "  --nmk <n>            Tiling: nmk (default: 32)\n"
              << "  --help               Show this help\n";
}

std::vector<std::string> split_string(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::size_t start = 0;
    std::size_t end = s.find(delim);
    while (end != std::string::npos) {
        result.push_back(s.substr(start, end - start));
        start = end + 1;
        end = s.find(delim, start);
    }
    result.push_back(s.substr(start));
    return result;
}

std::vector<std::uint32_t> parse_sizes(const std::string& s) {
    std::vector<std::uint32_t> sizes;
    for (const auto& part : split_string(s, ',')) {
        sizes.push_back(static_cast<std::uint32_t>(std::stoul(part)));
    }
    return sizes;
}

std::vector<bench::MatrixType> parse_types(const std::string& s) {
    std::vector<bench::MatrixType> types;
    for (const auto& part : split_string(s, ',')) {
        types.push_back(bench::parse_matrix_type(part));
    }
    return types;
}

} // namespace

int main(int argc, char* argv[]) {
    bench::BenchConfig config;
    config.impl = IMPL_NAME;
    config.runs = 100;
    config.warmup = 5;
    config.pause_ms = 50;
    config.device = "unknown";
    config.output_file = "results.csv";
    config.tiling = {64, 64, 128, 32, 32};

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            config.sizes = parse_sizes(argv[++i]);
        } else if (std::strcmp(argv[i], "--types") == 0 && i + 1 < argc) {
            config.matrix_types = parse_types(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            config.runs = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            config.warmup = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--pause-ms") == 0 && i + 1 < argc) {
            config.pause_ms = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            config.device = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (std::strcmp(argv[i], "--mblk") == 0 && i + 1 < argc) {
            config.tiling.mblk = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--nblk") == 0 && i + 1 < argc) {
            config.tiling.nblk = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--kblk") == 0 && i + 1 < argc) {
            config.tiling.kblk = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--mmk") == 0 && i + 1 < argc) {
            config.tiling.mmk = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (std::strcmp(argv[i], "--nmk") == 0 && i + 1 < argc) {
            config.tiling.nmk = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else {
            std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate
    if (config.sizes.empty()) {
        std::cerr << "Error: --sizes is required\n";
        return 1;
    }
    if (config.matrix_types.empty()) {
        config.matrix_types = {bench::MatrixType::RandomDense};
    }

    std::cout << "Benchmark Configuration:\n"
              << "  Implementation: " << config.impl << "\n"
              << "  Device: " << config.device << "\n"
              << "  Sizes: ";
    for (std::size_t i = 0; i < config.sizes.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << config.sizes[i];
    }
    std::cout << "\n  Matrix types: ";
    for (std::size_t i = 0; i < config.matrix_types.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << bench::matrix_type_name(config.matrix_types[i]);
    }
    std::cout << "\n  Runs: " << config.runs
              << "\n  Warmup: " << config.warmup
              << "\n  Pause: " << config.pause_ms << " ms\n"
              << "  Tiling: mblk=" << config.tiling.mblk
              << ", nblk=" << config.tiling.nblk
              << ", kblk=" << config.tiling.kblk
              << ", mmk=" << config.tiling.mmk
              << ", nmk=" << config.tiling.nmk << "\n"
              << "  Output: " << config.output_file << "\n\n";

    // Open CSV file
    bench::CsvWriter csv(config.output_file);
    csv.write_header();

    // Random number generator with fixed seed for reproducibility
    std::mt19937 rng(42);

    // Run benchmarks
    for (std::uint32_t size : config.sizes) {
        const std::uint32_t m = size;
        const std::uint32_t n = size;
        const std::uint32_t k = size;

        std::cout << "Size " << size << "x" << size << "x" << size << ":\n";

        for (bench::MatrixType type : config.matrix_types) {
            std::cout << "  " << bench::matrix_type_name(type) << ": " << std::flush;

            // Allocate matrices
            const std::uint32_t rowBytesA = n / 8;
            const std::uint32_t colBytesB = n / 8;

            auto Apos = std::make_unique<std::uint8_t[]>(static_cast<std::size_t>(m) * rowBytesA);
            auto Aneg = std::make_unique<std::uint8_t[]>(static_cast<std::size_t>(m) * rowBytesA);
            auto B = std::make_unique<std::uint8_t[]>(static_cast<std::size_t>(k) * colBytesB);

            // Warmup runs
            for (std::uint32_t w = 0; w < config.warmup; ++w) {
                bench::generate_ternary_matrix(Apos.get(), Aneg.get(), m, n, type, rng);
                bench::generate_binary_matrix(B.get(), n, k, rng);

#if defined(USE_IMPL_02)
                auto result = GeMMCoded(Apos.get(), Aneg.get(), B.get(), m, n, k);
                delete[] result.first;
                delete[] result.second;
#elif defined(USE_IMPL_03) || defined(USE_IMPL_04)
                TilingParams tp = {config.tiling.mblk, config.tiling.nblk, config.tiling.kblk,
                                   config.tiling.mmk, config.tiling.nmk};
                std::int32_t* result = GemmTBN_Blocked(Apos.get(), Aneg.get(), B.get(), m, n, k, tp);
                delete[] result;
#else
                tbn::TernaryMatrixView Aview{
                    std::span<const std::uint8_t>(Apos.get(), static_cast<std::size_t>(m) * rowBytesA),
                    std::span<const std::uint8_t>(Aneg.get(), static_cast<std::size_t>(m) * rowBytesA),
                    m, n
                };
                tbn::BinaryMatrixView Bview{
                    std::span<const std::uint8_t>(B.get(), static_cast<std::size_t>(k) * colBytesB),
                    n, k
                };
                tbn::TilingParams tp = {config.tiling.mblk, config.tiling.nblk, config.tiling.kblk,
                                        config.tiling.mmk, config.tiling.nmk};
                tbn::GemmEngine engine;
                auto result = engine.compute(Aview, Bview, tp);
                (void)result;
#endif
                bench::sleep_ms(config.pause_ms);
            }

            // Actual benchmark runs
            for (std::uint32_t run = 1; run <= config.runs; ++run) {
                // Generate new random matrices for each run
                bench::generate_ternary_matrix(Apos.get(), Aneg.get(), m, n, type, rng);
                bench::generate_binary_matrix(B.get(), n, k, rng);

                bench::Timer timer;
                timer.start();

#if defined(USE_IMPL_02)
                auto result = GeMMCoded(Apos.get(), Aneg.get(), B.get(), m, n, k);
                timer.stop();
                delete[] result.first;
                delete[] result.second;
#elif defined(USE_IMPL_03) || defined(USE_IMPL_04)
                TilingParams tp = {config.tiling.mblk, config.tiling.nblk, config.tiling.kblk,
                                   config.tiling.mmk, config.tiling.nmk};
                std::int32_t* result = GemmTBN_Blocked(Apos.get(), Aneg.get(), B.get(), m, n, k, tp);
                timer.stop();
                delete[] result;
#else
                tbn::TernaryMatrixView Aview{
                    std::span<const std::uint8_t>(Apos.get(), static_cast<std::size_t>(m) * rowBytesA),
                    std::span<const std::uint8_t>(Aneg.get(), static_cast<std::size_t>(m) * rowBytesA),
                    m, n
                };
                tbn::BinaryMatrixView Bview{
                    std::span<const std::uint8_t>(B.get(), static_cast<std::size_t>(k) * colBytesB),
                    n, k
                };
                tbn::TilingParams tp = {config.tiling.mblk, config.tiling.nblk, config.tiling.kblk,
                                        config.tiling.mmk, config.tiling.nmk};
                tbn::GemmEngine engine;
                auto result = engine.compute(Aview, Bview, tp);
                timer.stop();
                (void)result;
#endif

                const double time_ms = timer.elapsed_ms();
                const double gflops = bench::calculate_gflops(m, n, k, time_ms);

                csv.write_row(
                    config.device, config.impl, bench::matrix_type_name(type),
                    m, n, k, run, time_ms, gflops,
                    config.tiling.mblk, config.tiling.nblk, config.tiling.kblk,
                    config.tiling.mmk, config.tiling.nmk
                );

                if (run % 10 == 0) {
                    std::cout << "." << std::flush;
                }

                bench::sleep_ms(config.pause_ms);
            }
            std::cout << " done\n";
        }
    }

    std::cout << "\nBenchmark complete. Results saved to: " << config.output_file << "\n";
    return 0;
}
