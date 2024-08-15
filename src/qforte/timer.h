#ifndef _timer_h_
#define _timer_h_

#include <chrono>
#include <vector>
#include <string>
#include <unordered_map>
/**
 * @brief A timer class that returns the elapsed time
 */
class local_timer {
  public:
    local_timer() : start_(std::chrono::high_resolution_clock::now()) {}

    /// reset the timer
    void reset();

    /// return the elapsed time in seconds
    double get();

    /// return the elapsed time in seconds
    void record(std::string name);

    void acc_record(const std::string& name);

    /// returns a string representing a timings table
    std::string str_table(); 

    std::string acc_str_table();

    std::unordered_map<std::string, double> get_acc_timings() { return acc_timings_; }

    std::vector<std::pair<std::string, double>> get_timings() { return timings_; }



  private:
    /// stores the time when this object is created
    std::chrono::high_resolution_clock::time_point start_;

    /// stores the timings and names
    std::vector<std::pair<std::string, double>> timings_;

    std::unordered_map<std::string, double> acc_timings_;

    
};
#endif // _timer_h_
