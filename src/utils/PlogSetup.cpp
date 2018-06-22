#include <utils/PlogSetup.h>

namespace CTL::util {

PlogSetup::PlogSetup(plog::Severity verbosityLevel, std::string csvLogFile, bool logToConsole)
    : verbosityLevel(verbosityLevel)
    , csvLogFile(csvLogFile)
    , logToConsole(logToConsole)
{
}

void PlogSetup::initLogging()
{
    if (!csvLogFile.empty()) {
        static plog::RollingFileAppender<plog::CsvFormatter> fileAppender(csvLogFile.c_str(), 0, 0);
        if (logToConsole) {
            static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
            plog::init(verbosityLevel, &fileAppender).addAppender(&consoleAppender);
        } else {
            plog::init(verbosityLevel, &fileAppender);
        }
    } else {
        if (logToConsole) {
            static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
            plog::init(verbosityLevel, &consoleAppender);
        }
    }
    return;
}

} //namespace CTL::util
