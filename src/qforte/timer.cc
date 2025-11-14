// #include <timer.h>
// #include <string>
// #include <iostream>
// #include <iomanip>
// #include <sstream>

// void local_timer::reset() {
//      start_ = std::chrono::high_resolution_clock::now();
// }

// double local_timer::get() {
//     auto duration = std::chrono::high_resolution_clock::now() - start_;
//     return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
// }

// void local_timer::record(std::string name) {
//     auto duration = std::chrono::high_resolution_clock::now() - start_;
//     double time = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
//     timings_.push_back(std::make_pair(name, time));
// }

// void local_timer::acc_record(const std::string& name) {
//     auto duration = std::chrono::high_resolution_clock::now() - start_;
//     double time = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
//     if (acc_timings_.find(name) != acc_timings_.end()) {
//         acc_timings_[name] += time;
//     } else {
//         acc_timings_[name] = time;
//     }

// }

// std::string local_timer::str_table() {
//     std::stringstream result;

//     size_t max = 0;
//     double total_time = 0.0;

//     for (const auto& entry : timings_) {
//         max = std::max(max, entry.first.length());
//         total_time += entry.second;
//     }

//     max = std::max(max, static_cast<size_t>(10));

//     result << std::setw(max) << "Process name" << std::setw(max) << "Time (s)" << std::setw(max) << "Percent" << "\n";
//     result << std::setw(max) << "=============" << std::setw(max) << "=============" << std::setw(max) << "=============" << "\n";

//     for (const auto& entry : timings_) {
//         double percent = (entry.second / total_time) * 100.0;
//         result << std::setw(max) << entry.first << std::fixed << std::setprecision(4) << std::setw(max) << entry.second
//                << std::fixed << std::setprecision(2) << std::setw(max) << percent << "\n";
//     }

//     result << "\n";

//     result << std::setw(max) << "Total Time" << std::fixed << std::setprecision(4) << std::setw(max) << total_time
//            << std::fixed << std::setprecision(2) << std::setw(max) << 100.0 << "\n";

//     return result.str();
// }

// std::string local_timer::acc_str_table() {
//     std::stringstream result;

//     size_t max = 0;
//     double total_time = 0.0;

//     for (const auto& entry : acc_timings_) {
//         max = std::max(max, entry.first.length());
//         total_time += entry.second;
//     }

//     max = std::max(max, static_cast<size_t>(10));

//     result << std::setw(max) << "Process name" << std::setw(max) << "Time (s)" << std::setw(max) << "Percent" << "\n";
//     result << std::setw(max) << "=============" << std::setw(max) << "=============" << std::setw(max) << "=============" << "\n";

//     for (const auto& entry : acc_timings_) {
//         double percent = (entry.second / total_time) * 100.0;
//         result << std::setw(max) << entry.first << std::fixed << std::setprecision(4) << std::setw(max) << entry.second
//                << std::fixed << std::setprecision(2) << std::setw(max) << percent << "\n";
//     }

//     result << "\n";

//     result << std::setw(max) << "Total Time" << std::fixed << std::setprecision(4) << std::setw(max) << total_time
//            << std::fixed << std::setprecision(2) << std::setw(max) << 100.0 << "\n";

//     return result.str();
// }

#include "timer.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

void local_timer::reset() {
    start_ = clock::now();
}

double local_timer::get() {
    auto duration = clock::now() - start_;
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

void local_timer::record(std::string name) {
    double time = get();
    timings_.emplace_back(std::move(name), time);
}

void local_timer::acc_record(const std::string& name) {
    double time = get();
    acc_record_seconds(name, time);
}

void local_timer::acc_record_seconds(const std::string& name, double seconds) {
    auto it = acc_timings_.find(name);
    if (it != acc_timings_.end()) {
        it->second += seconds;
    } else {
        acc_timings_.emplace(name, seconds);
    }
}

// ------- new explicit nesting-safe sections --------

void local_timer::acc_begin(const std::string& name) {
    active_starts_[name].push_back(clock::now());
}

void local_timer::acc_end(const std::string& name) {
    auto it = active_starts_.find(name);
    if (it == active_starts_.end() || it->second.empty()) {
        // Mismatched stop; nothing to do. You could throw or log if desired.
        return;
    }
    const time_point t0 = it->second.back();
    it->second.pop_back();

    const auto dur   = clock::now() - t0;
    const double sec = std::chrono::duration_cast<std::chrono::duration<double>>(dur).count();

    // Add a per-call row (non-accumulating) and also add to accumulated totals
    timings_.emplace_back(name, sec);
    acc_record_seconds(name, sec);
}

void local_timer::accumulate(std::string name) {
    // Calculate the elapsed time
    double elapsed_time = get();

    // Check if the task name already exists in the timings_ vector
    for (auto& timing : timings_) {
        if (timing.first == name) {
            // If it exists, add the elapsed time to the existing total
            timing.second += elapsed_time;
            return;
        }
    }

    // If the task name does not exist, add a new entry
    timings_.emplace_back(name, elapsed_time);
}

std::string local_timer::str_table() {
    std::stringstream result;

    size_t max = 0;
    double total_time = 0.0;

    for (const auto& entry : timings_) {
        max = std::max(max, entry.first.length());
        total_time += entry.second;
    }

    max = std::max(max, static_cast<size_t>(12));

    result << std::setw(max) << "Process name"
           << std::setw(max) << "Time (s)"
           << std::setw(max) << "Percent" << "\n";

    result << std::setw(max) << "============="
           << std::setw(max) << "========="
           << std::setw(max) << "=======" << "\n";

    for (const auto& entry : timings_) {
        double percent = (total_time > 0.0) ? (entry.second / total_time) * 100.0 : 0.0;
        result << std::setw(max) << entry.first
               << std::fixed << std::setprecision(4) << std::setw(max) << entry.second
               << std::fixed << std::setprecision(2) << std::setw(max) << percent << "\n";
    }

    result << "\n";
    result << std::setw(max) << "Total Time"
           << std::fixed << std::setprecision(4) << std::setw(max) << total_time
           << std::fixed << std::setprecision(2) << std::setw(max) << 100.0 << "\n";

    return result.str();
}

std::string local_timer::acc_str_table() {
    std::stringstream result;

    size_t max = 0;
    double total_time = 0.0;

    for (const auto& entry : acc_timings_) {
        max = std::max(max, entry.first.length());
        total_time += entry.second;
    }

    max = std::max(max, static_cast<size_t>(12));

    result << std::setw(max) << "Process name"
           << std::setw(max) << "Time (s)"
           << std::setw(max) << "Percent" << "\n";

    result << std::setw(max) << "============="
           << std::setw(max) << "========="
           << std::setw(max) << "=======" << "\n";

    for (const auto& entry : acc_timings_) {
        double percent = (total_time > 0.0) ? (entry.second / total_time) * 100.0 : 0.0;
        result << std::setw(max) << entry.first
               << std::fixed << std::setprecision(4) << std::setw(max) << entry.second
               << std::fixed << std::setprecision(2) << std::setw(max) << percent << "\n";
    }

    result << "\n";
    result << std::setw(max) << "Total Time"
           << std::fixed << std::setprecision(4) << std::setw(max) << total_time
           << std::fixed << std::setprecision(2) << std::setw(max) << 100.0 << "\n";

    return result.str();
}




