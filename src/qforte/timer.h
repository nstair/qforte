// #ifndef _timer_h_
// #define _timer_h_

// #include <chrono>
// #include <vector>
// #include <string>
// #include <unordered_map>
// /**
//  * @brief A timer class that returns the elapsed time
//  */
// class local_timer {
//   public:
//     local_timer() : start_(std::chrono::high_resolution_clock::now()) {}

//     /// reset the timer
//     void reset();

//     /// return the elapsed time in seconds
//     double get();

//     /// return the elapsed time in seconds
//     void record(std::string name);

//     void acc_record(const std::string& name);

//     /// returns a string representing a timings table
//     std::string str_table(); 

//     std::string acc_str_table();

//     std::unordered_map<std::string, double> get_acc_timings() { return acc_timings_; }

//     std::vector<std::pair<std::string, double>> get_timings() { return timings_; }



//   private:
//     /// stores the time when this object is created
//     std::chrono::high_resolution_clock::time_point start_;

//     /// stores the timings and names
//     std::vector<std::pair<std::string, double>> timings_;

//     std::unordered_map<std::string, double> acc_timings_;

    
// };
// #endif // _timer_h_

#ifndef _timer_h_
#define _timer_h_

#include <chrono>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @brief A timer class that returns the elapsed time
 *
 * Existing APIs preserved:
 *   - reset(), get(), record(name), acc_record(name),
 *     str_table(), acc_str_table(), get_acc_timings(), get_timings()
 *
 * New explicit, nesting-safe section timing:
 *   - acc_begin(name) / acc_end(name)
 *   - accumulate_start(name) / accumulate_stop(name)  (aliases)
 *   - acc_record_seconds(name, seconds)
 */
class local_timer {
public:
    using clock      = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

    local_timer() : start_(clock::now()) {}

    /// reset the epoch used by get()/record()/acc_record()
    void reset();

    /// return the elapsed time in seconds since last reset() (or construction)
    double get();

    /// record elapsed time since last reset() into the non-accumulating table
    void record(std::string name);

    /// accumulate elapsed time since last reset() into the accumulating table
    void acc_record(const std::string& name);

    /// accumulate an explicit number of seconds into the accumulating table
    void acc_record_seconds(const std::string& name, double seconds);

    /// Start/stop explicit accumulating sections (nesting-safe, inclusive timing)
    void acc_begin(const std::string& name);                 // start a named section
    void acc_end(const std::string& name);                   // stop the most recent same-named section

    // Friendly aliases the way you asked to call them
    // inline void accumulate_start(const std::string& name) { acc_begin(name); }
    // inline void accumulate_stop(const std::string& name) { acc_end(name);   }

    /// returns a string representing a timings table (non-accumulating, per call)
    std::string str_table();

    /// returns a string representing the accumulating timings table (totals per name)
    std::string acc_str_table();

    std::unordered_map<std::string, double> get_acc_timings() { return acc_timings_; }
    std::vector<std::pair<std::string, double>> get_timings() { return timings_; }

private:
    // epoch for legacy “since reset()” methods
    time_point start_;

    // non-accumulating records (one row per record() or acc_end())
    std::vector<std::pair<std::string, double>> timings_;

    // accumulating records (sum over all acc_* per name)
    std::unordered_map<std::string, double> acc_timings_;

    // per-name stack of start times to support nested/overlapping sections
    std::unordered_map<std::string, std::vector<time_point>> active_starts_;
};

#endif // _timer_h_

