#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <nvml.h>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <csignal>

#define BUFFER_SIZE 1000 // number of rows before flushing data to CSV


// For error checking
#ifndef NVML_RT_CALL
#define NVML_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<nvmlReturn_t>( call );                                                               \
        if ( status != NVML_SUCCESS )                                                                                  \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed "                                      \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     nvmlErrorString( status ),                                                                        \
                     status );                                                                                         \
    }
#endif

class GpuMonitor {
    private:
        typedef struct _stats {
            std::time_t             timestamp;
            uint                    temperature;
            uint                    clockSM;
            uint                    clockGraphics;
            uint                    clockMem;
            uint                    clockMemMax;
            uint                    powerUsage;
            uint                    powerLimit;
            long long unsigned int  energy;
            nvmlUtilization_t       gpuUtil;
            nvmlMemory_t            memoryStats;
            nvmlPstates_t           pstate;
            unsigned long long      throttleReasons;
        } stats;

        std::vector<std::string> headerNames_ = {
            "timestamp",
            "temp",
            "clock_sm_mhz",
            "clock_graphics_mhz",
            "clock_mem_mhz",
            "clock_mem_max_mhz",
            "power_draw_mW",
            "power_limit_mW",
            "energy_draw_j",
            "util_sm",
            "util_mem",
            "mem_used_bytes",
            "mem_free_bytes",
            "pstate",
            "clock_throttle_reason_active"
        };

        std::vector<stats> buffer_;
        std::string filename_;
        std::ofstream logFile_;
        nvmlDevice_t device_;
        static volatile sig_atomic_t continueMonitoring_;

        void writeHeaders() {
            for (int i = 0; i < (static_cast<int>(headerNames_.size( ))-1); i++)
                logFile_ << headerNames_[i] << ",";
            // Leave off last commma
            logFile_ << headerNames_[static_cast<int>(headerNames_.size())- 1];
            logFile_ << "\n";
        }

        void flushBuffer() {
            // Flush buffer to CSV
            for (int i=0; i < static_cast<int>(buffer_.size( )); i++) {
                logFile_ << buffer_[i].timestamp << ","
                         << buffer_[i].temperature << ","
                         << buffer_[i].clockSM << ","
                         << buffer_[i].clockGraphics << ","
                         << buffer_[i].clockMem << ","
                         << buffer_[i].clockMemMax << ","
                         << buffer_[i].powerUsage  << ","
                         << buffer_[i].powerLimit << ","
                         << buffer_[i].energy << ","
                         << buffer_[i].gpuUtil.gpu << ","
                         << buffer_[i].gpuUtil.memory << ","
                         << buffer_[i].memoryStats.used << ","
                         << buffer_[i].memoryStats.free << ","
                         << buffer_[i].pstate << ","
                         << buffer_[i].throttleReasons << "\n";
            }
            logFile_.flush();
            buffer_.clear();
        }

        static void signalHandler(int signum) {
            continueMonitoring_ = 0;
        }
        void collectDataSample(stats& device_stats) {
            device_stats.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            NVML_RT_CALL(nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &device_stats.temperature));
            NVML_RT_CALL(nvmlDeviceGetClock(device_, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &device_stats.clockSM));
            NVML_RT_CALL(nvmlDeviceGetClock(device_, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_APP_CLOCK_TARGET, &device_stats.clockGraphics));
            NVML_RT_CALL(nvmlDeviceGetClock(device_, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &device_stats.clockMem));
            NVML_RT_CALL(nvmlDeviceGetClock(device_, NVML_CLOCK_MEM, NVML_CLOCK_ID_APP_CLOCK_TARGET, &device_stats.clockMemMax));
            NVML_RT_CALL(nvmlDeviceGetPowerUsage(device_, &device_stats.powerUsage));
            NVML_RT_CALL(nvmlDeviceGetEnforcedPowerLimit(device_, &device_stats.powerLimit));
            NVML_RT_CALL(nvmlDeviceGetTotalEnergyConsumption(device_, &device_stats.energy));
            // NVML_RT_CALL(nvmlDeviceGetUtilizationRates(device_, &device_stats.gpuUtil));
            NVML_RT_CALL(nvmlDeviceGetMemoryInfo(device_, &device_stats.memoryStats));
            NVML_RT_CALL(nvmlDeviceGetPerformanceState( device_, &device_stats.pstate));
            NVML_RT_CALL(nvmlDeviceGetCurrentClocksThrottleReasons(device_, &device_stats.throttleReasons));
            
            buffer_.push_back(device_stats);

            // Flush buffer when full
            if (buffer_.size() >= BUFFER_SIZE) {
                flushBuffer();
            }
        }

    public:
        // Constructor
        GpuMonitor(int const gpuIndex, std::string const filename) {

            // Register signal handler
            std::signal(SIGINT, signalHandler);

            // Open CSV file and write headers to it
            logFile_.open(filename, std::ios::app);
            if (!logFile_.is_open()) {
                throw std::runtime_error("Failed to open log file: " + filename);
            }
            writeHeaders();

            // Iniitialize NVML
            NVML_RT_CALL(nvmlInit());

            // Get GPU handle
            NVML_RT_CALL(nvmlDeviceGetHandleByIndex(gpuIndex, &device_));

            // Reserve memory for buffer storing data
            buffer_.reserve(BUFFER_SIZE);
        }

        // Destructor
        ~GpuMonitor() {
            flushBuffer();
            NVML_RT_CALL(nvmlShutdown());
            logFile_.close();
        }

        // Start monitoring
        void monitor(int monitorInterval) {
            stats device_stats {};
            while (continueMonitoring_) {
                collectDataSample(device_stats);
                std::this_thread::sleep_for(std::chrono::milliseconds(monitorInterval));
            }

            // Gracefully exiting, get 5 more seconds worth of data to let things stablize
            auto startTime = std::chrono::high_resolution_clock::now();
            while (std::chrono::high_resolution_clock::now() - startTime < std::chrono::seconds(2)) {
                collectDataSample(device_stats);
                std::this_thread::sleep_for(std::chrono::milliseconds(monitorInterval));
            }
            flushBuffer();

        }

};

volatile sig_atomic_t GpuMonitor::continueMonitoring_ = 1;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <gpu_index (0-3)> <filename>\n";
        return 1;
    }

    int gpuIndex = std::atoi(argv[1]);
    if (gpuIndex < 0 || gpuIndex > 3) {
        std::cerr << "Invalid GPU index. Choose between 0 and 3.\n";
        return 1;
    }

    std::string filename = argv[2];

    // std::string filename = "gpu" + std::to_string(gpuIndex) + "_log.csv";

    try {
        GpuMonitor monitor(gpuIndex, filename);
        monitor.monitor(5);  // Monitor at 5ms intervals
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

