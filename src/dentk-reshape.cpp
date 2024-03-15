// The purpose of this tool is to filter out outer bone structures.
// Logging
#include "PLOG/PlogSetup.h"

// External libraries
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <iostream>
#include <regex>
#include <string>

// External libraries
#include "CLI/CLI.hpp" //Command line parser

// Internal libraries
#include "BufferedFrame2D.hpp"
#include "DEN/DenAsyncFrame2DWritter.hpp"
#include "DEN/DenFileInfo.hpp"
#include "DEN/DenFrame2DReader.hpp"
#include "Frame2DReaderI.hpp"
#include "PROG/ArgumentsForce.hpp"
#include "PROG/ArgumentsFramespec.hpp"
#include "PROG/Program.hpp"

using namespace KCT;
using namespace KCT::util;

// class declarations
// class declarations
class Args : public ArgumentsFramespec, public ArgumentsForce
{
    void defineArguments();
    int postParse();
    int preParse() { return 0; };

public:
    Args(int argc, char** argv, std::string prgName)
        : Arguments(argc, argv, prgName)
        , ArgumentsFramespec(argc, argv, prgName)
        , ArgumentsForce(argc, argv, prgName){};
    std::string input_den = "";
    std::string output_den = "";
    uint32_t dimx_after;
    uint32_t dimy_after;
    uint32_t dimz_after;
};

template <typename T>
void addToFrame(std::shared_ptr<io::Frame2DI<T>> to,
                std::shared_ptr<io::Frame2DI<T>> from,
                double weight)
{
    uint32_t dimx = from->dimx();
    uint32_t dimy = from->dimy();
    if(dimx != to->dimx() || dimy != to->dimy())
    {
        std::string err = io::xprintf("There is dimension mismatch between frames!");
        LOGE << err;
        throw std::runtime_error(err);
    }
    T val;
    for(uint32_t i = 0; i != dimx; i++)
    {
        for(uint32_t j = 0; j != dimy; j++)
        {
            val = to->get(i, j);
            to->set(val + weight * from->get(i, j), i, j);
        }
    }
}

template <typename T>
/**
 * @brief From given frame with dimensions dimx dimy creates reshaped frame with dimensions
 * dimx_after dimy_after
 * indexIncrease should be equal to count+weightOfFirstElement+weightOfLastElement
 *
 * @param f
 * @param dimx_after
 * @param dimy_after
 *
 * @return
 */
std::shared_ptr<io::Frame2DI<T>>
averageFrames(const std::deque<std::shared_ptr<io::Frame2DI<T>>>& Q,
              double weightOfFirstElement,
              double weightOfLastElement,
              double indexIncrease,
              uint32_t count)
{
    if(count < 2)
    {
        std::string err = io::xprintf(
            "Count of frames in the queue to add together should be greater than 1 but it is %d!",
            count);
        LOGE << err;
        throw std::runtime_error(err);
    }
    uint32_t dimx = Q[0]->dimx();
    uint32_t dimy = Q[0]->dimy();
    std::shared_ptr<io::BufferedFrame2D<T>> g
        = std::make_shared<io::BufferedFrame2D<T>>(T(0), dimx, dimy);
    addToFrame<T>(g, Q[0], weightOfFirstElement / indexIncrease);
    for(std::size_t i = 1; i + 1 < count; i++)
    {
        addToFrame<T>(g, Q[i], 1.0 / indexIncrease);
    }
    addToFrame<T>(g, Q[count - 1], weightOfLastElement / indexIncrease);
    return g;
}

template <typename T>
/**
 * @brief From given frame with dimensions dimx dimy creates reshaped frame with dimensions
 * dimx_after dimy_after
 *
 * @param f
 * @param dimx_after
 * @param dimy_after
 *
 * @return
 */
std::shared_ptr<io::Frame2DI<T>>
createReshapedFrame(std::shared_ptr<io::Frame2DI<T>> f, uint32_t dimx_after, uint32_t dimy_after)
{
    uint32_t dimx = f->dimx();
    uint32_t dimy = f->dimy();
    if(dimx != dimx_after)
    {

        std::shared_ptr<io::BufferedFrame2D<T>> tmp
            = std::make_shared<io::BufferedFrame2D<T>>(T(0), dimx_after, dimy);

        double indexBegin, indexEnd, indexIncrease; // Indexes to array f
        uint32_t integerBegin, integerEnd;
        indexIncrease = double(dimx) / double(dimx_after);
        indexBegin = 0.0;
        indexEnd = indexIncrease;
        T sum = 0;
        for(uint32_t i = 0; i != dimx_after; i++)
        {
            integerBegin = (uint32_t)indexBegin;
            integerEnd = (uint32_t)indexEnd;
            for(uint32_t j = 0; j != dimy; j++)
            {
                sum = 0;
                if(integerBegin == integerEnd)
                {
                    tmp->set(f->get(integerBegin, j), i, j);
                } else
                {
                    sum += (1.0 + integerBegin - indexBegin) * f->get(integerBegin, j);
                    for(unsigned int q = integerBegin + 1; q != integerEnd; q++)
                    {
                        sum += f->get(q, j);
                    }
                    if(integerEnd != dimx) // That would happen in case of numerical error where
                                           // there is overflow
                    {
                        sum += (indexEnd - integerEnd) * f->get(integerEnd, j);
                    }
                    sum /= indexIncrease;
                    tmp->set(sum, i, j);
                }
            }
            indexBegin = indexEnd;
            indexEnd += indexIncrease;
        }
        return createReshapedFrame<T>(tmp, dimx_after, dimy_after);
    } else if(dimy != dimy_after)
    {

        std::shared_ptr<io::BufferedFrame2D<T>> g
            = std::make_shared<io::BufferedFrame2D<T>>(T(0), dimx_after, dimy_after);

        double indexBegin, indexEnd, indexIncrease; // Indexes to array f
        uint32_t integerBegin, integerEnd;
        indexIncrease = double(dimy) / double(dimy_after);
        indexBegin = 0.0;
        indexEnd = indexIncrease;
        T sum = 0;
        for(uint32_t j = 0; j != dimy_after; j++)
        {
            integerBegin = (uint32_t)indexBegin;
            integerEnd = (uint32_t)indexEnd;
            for(uint32_t i = 0; i != dimx_after; i++)
            {
                sum = 0;
                if(integerBegin == integerEnd)
                {
                    g->set(f->get(i, integerBegin), i, j);
                } else
                {
                    sum += (1.0 + integerBegin - indexBegin) * f->get(i, integerBegin);
                    for(unsigned int q = integerBegin + 1; q != integerEnd; q++)
                    {
                        sum += f->get(i, q);
                    }
                    if(integerEnd != dimy) // That would happen in case of numerical error where
                                           // there is overflow
                    {
                        sum += (indexEnd - integerEnd) * f->get(i, integerEnd);
                    }
                    sum /= indexIncrease;
                    g->set(sum, i, j);
                }
            }
            indexBegin = indexEnd;
            indexEnd += indexIncrease;
        }
        return g;
    } else
    {
        // when dimensions matchex
        return f;
    }
}

template <typename T>
void rescaleFiles(Args a)
{
    std::shared_ptr<io::Frame2DReaderI<T>> denReader
        = std::make_shared<io::DenFrame2DReader<T>>(a.input_den);
    std::shared_ptr<io::AsyncFrame2DWritterI<T>> outputWritter
        = std::make_shared<io::DenAsyncFrame2DWritter<T>>(a.output_den, a.dimx_after, a.dimy_after,
                                                          a.dimz_after);
    uint32_t dimz_input = denReader->getFrameCount();
    if(dimz_input == a.dimz_after)
    {
        for(uint32_t k = 0; k < a.dimz_after; k++)
        {
            outputWritter->writeFrame(
                *(createReshapedFrame<T>(denReader->readFrame(k), a.dimx_after, a.dimy_after)), k);
        }
        return;
    }
    double kIndexBegin, kIndexEnd, kIndexIncrease;
    uint32_t integerBegin, integerEnd;
    kIndexIncrease = double(dimz_input) / double(a.dimz_after);
    kIndexBegin = 0.0;
    kIndexEnd = kIndexIncrease;
    std::deque<std::shared_ptr<io::Frame2DI<T>>> Q; // Queue with the access to the middle elements
    Q.push_back(createReshapedFrame<T>(denReader->readFrame(0), a.dimx_after, a.dimy_after));
    uint32_t queueBeginIndex = 0;
    for(uint32_t k_output = 0; k_output < a.dimz_after; k_output++)
    {
        integerBegin = (uint32_t)kIndexBegin;
        integerEnd = (uint32_t)kIndexEnd;
        while(queueBeginIndex < integerBegin && Q.size() > 0)
        {
            Q.pop_front();
            queueBeginIndex++;
        }
        if(Q.size() == 0)
        {
            queueBeginIndex = integerBegin;
            Q.push_back(createReshapedFrame<T>(denReader->readFrame(integerBegin), a.dimx_after,
                                               a.dimy_after));
        }
        // How many elements I need to process
        uint32_t elementsInView = std::min(integerEnd + 1, dimz_input) - integerBegin;
        while(Q.size() < elementsInView)
        {
            Q.push_back(createReshapedFrame<T>(denReader->readFrame(integerBegin + Q.size()),
                                               a.dimx_after, a.dimy_after));
        }
        // Now we are about to average the frames
        std::shared_ptr<io::Frame2DI<T>> xf;
        if(elementsInView == 1)
        {
            xf = Q[0];
        } else
        {
            if(integerEnd == dimz_input)
            { // This should be due to numerical error so we don't care about integerEnd here
                xf = averageFrames<T>(Q, 1.0 + integerBegin - kIndexBegin, 1.0, kIndexIncrease,
                                      elementsInView);
            } else
            {
                xf = averageFrames<T>(Q, 1.0 + integerBegin - kIndexBegin, kIndexEnd - integerEnd,
                                      kIndexIncrease, elementsInView);
            }
        }
        outputWritter->writeFrame(*xf, k_output);
        kIndexBegin = kIndexEnd;
        kIndexEnd += kIndexIncrease;
    }
}

int main(int argc, char* argv[])
{
    Program PRG(argc, argv);
    // Argument parsing
    Args ARG(argc, argv, "Change the shape of the den file to a new dimensions.");
    int parseResult = ARG.parse();
    if(parseResult > 0)
    {
        return 0; // Exited sucesfully, help message printed
    } else if(parseResult != 0)
    {
        return -1; // Exited somehow wrong
    }
    PRG.startLog(true);
    io::DenFileInfo di(ARG.input_den);
    io::DenSupportedType dataType = di.getElementType();
    switch(dataType)
    {
    case io::DenSupportedType::UINT16: {
        rescaleFiles<uint16_t>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT32: {
        rescaleFiles<float>(ARG);
        break;
    }
    case io::DenSupportedType::FLOAT64: {
        rescaleFiles<double>(ARG);
        break;
    }
    default: {
        std::string errMsg = io::xprintf("Unsupported data type %s.",
                                         io::DenSupportedTypeToString(dataType).c_str());
        KCTERR(errMsg);
    }
    }
    PRG.endLog();
}

void Args::defineArguments()
{
    cliApp->add_option("input_den", input_den, "Input file.")->check(CLI::ExistingFile)->required();
    cliApp->add_option("output_den", output_den, "Output file.")->required();
    addForceArgs();
    cliApp->add_option("--dimx", dimx_after, "X dimension of the output file after reshape.");
    cliApp->add_option("--dimy", dimy_after, "Y dimension of the output file after reshape.");
    cliApp->add_option("--dimz", dimz_after, "Z dimension of the output file after reshape.");
}

int Args::postParse()
{
    if(!force)
    {
        if(io::pathExists(output_den))
        {
            LOGE << "Error: output file already exists, use -f to force overwrite.";
            return 1;
        }
    }
    io::DenFileInfo inf(input_den);
    if(cliApp->count("--dimx") == 0)
    {
        dimx_after = inf.dimx();
    }
    if(cliApp->count("--dimy") == 0)
    {
        dimy_after = inf.dimy();
    }
    if(cliApp->count("--dimz") == 0)
    {
        dimz_after = inf.dimz();
    }
    return 0;
}
