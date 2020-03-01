#include <utils/loguru.hpp>
#include <iostream>

#include <chrono>
#include <thread>

inline void sleep_ms(int ms)
{
	VLOG_F(2, "Sleeping for %d ms", ms);
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

inline void complex_calculation()
{
	LOG_SCOPE_F(INFO, "complex_calculation");
	LOG_F(INFO, "Starting time machine...");
	sleep_ms(200);
	LOG_F(WARNING, "The flux capacitor is not getting enough power!");
	sleep_ms(400);
	LOG_F(INFO, "Lighting strike!");
	VLOG_F(1, "Found 1.21 gigawatts...");
	sleep_ms(400);
	std::thread([]() {
		loguru::set_thread_name("the past");
		LOG_F(ERROR, "We ended up in 1985!");
	}).join();
	
}

int main(int argc, char *argv[])
{
	// Optional, but useful to time-stamp the start of the log.
    // Will also detect verbosity level on command line as -v.
	//loguru::init(argc, argv);
	

	loguru::add_file("logs/log.log", loguru::Append, loguru::Verbosity_MAX);

	//char log_path[64];
	//loguru::suggest_log_path("output/logs/", log_path, sizeof(log_path));
	//loguru::add_file(log_path, loguru::FileMode::Truncate, loguru::Verbosity_MAX);

	
	LOG_F(INFO, "Hello from main.cpp!");
	LOG_F(WARNING, "Example warning");
	LOG_F(ERROR, "Example error");
	LOG_F(INFO, "Try 5  lines: \n line 1 \n line 2\n line 3\n line 4\n line 5 ");
	complex_calculation();
	LOG_F(INFO, "main function about to end!");
}
