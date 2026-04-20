#include "../../include/tbn/utils/logging.hpp"

namespace tbn {

// Static member definitions
LogLevel Logger::global_level_ = LogLevel::INFO;
std::mutex Logger::mutex_;

void Logger::set_global_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    global_level_ = level;
}

LogLevel Logger::get_global_level() {
    std::lock_guard<std::mutex> lock(mutex_);
    return global_level_;
}

} // namespace tbn