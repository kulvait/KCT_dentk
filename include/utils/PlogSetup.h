#ifndef PLOGSETUP_H
#define PLOGSETUP_H

//External dependencies
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Log.h>
#include <string>

namespace CTL {
namespace util {

    class PlogSetup {
    public:
        PlogSetup(plog::Severity verbosityLevel, std::string csvLogFile, bool logToConsole);
        void initLogging();

    private:
        plog::Severity verbosityLevel; //Set to debug to see the debug messages, info messages
        std::string csvLogFile; //Set to empty string to disable
        bool logToConsole;
    };
}
} //namespace CTL::util
#endif //PLOGSETUP_H
