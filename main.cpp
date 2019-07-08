#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <queue>
#include <mutex>

/*
 ******************************************************************************
 * Picture to gray converter
 ******************************************************************************
*/

std::mutex gray_mutex;
struct GrayPortion {
    uint8_t* source;
    double* dest;
    int size;
};
std::queue<GrayPortion> portions;

void createGrayPortion() {
    GrayPortion portion;
    while (true) {
        {
            std::lock_guard<std::mutex> l{gray_mutex};
            if (portions.size() == 0) {
                return;
            }
            portion = portions.front();
            portions.pop();
        }
    
        for (int i = 0; i < portion.size; ++i) {
            *portion.dest = double(*portion.source + *(portion.source + 1) + *(portion.source + 2)) / 3.0;
            portion.dest++;
            portion.source += 3;
        }
    }
}

void createGrayMat(cv::Mat& pic, cv::Mat& gray, int threadCount)
{
    const int PORTION_SIZE = 65536;
    gray.create(pic.rows, pic.cols, CV_64FC1);
    size_t size = pic.rows * pic.cols;
    for (int i = 0; i < size; i += PORTION_SIZE) {
        int portionSize = PORTION_SIZE;
        if (i + PORTION_SIZE >= size) {
            portionSize = size - i;
        }
        portions.push({
            pic.data + i * 3,
            (double*)gray.data + i,
            portionSize
        });
    }
    std::vector<std::thread> threads;
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back(createGrayPortion);
    }
    for (auto& t : threads) {
        t.join();
    }
}

/*
 ******************************************************************************
 * Blur converter
 ******************************************************************************
*/

const double BLUR_DEVIATION = 20;
const int BLUR_VECTOR_WING = ceil(3 * BLUR_DEVIATION);
const int BLUR_VECTOR_DIMENTION = BLUR_VECTOR_WING * 2 + 1;
double blurVector[BLUR_VECTOR_DIMENTION];

void fillBlurVector()
{
    for (int i = 0; i <= BLUR_VECTOR_WING; ++i) {
        blurVector[BLUR_VECTOR_WING + i] = 
            blurVector[BLUR_VECTOR_WING - i] = 
            (1.0 / sqrt(2.0 * M_PI * BLUR_DEVIATION)) * exp(- i * i / (BLUR_DEVIATION * BLUR_DEVIATION));
    }
    double sum = 0.0;
    for (const auto& el : blurVector) {
        sum += el;
    }
    for (auto& el : blurVector) {
        el /= sum;
    }
}

std::mutex blur_mutex;
struct BlurPortion {
    double* source;
    double* dest;
    int size;
    int step;
};

std::queue<BlurPortion> blurPortions;

void createBlurPortion()
{
    BlurPortion portion;
    while (true) {
        {
            std::lock_guard<std::mutex> l{blur_mutex};
            if (blurPortions.size() == 0) {
                return;
            }
            portion = blurPortions.front();
            blurPortions.pop();
        }

        int beginPart = BLUR_VECTOR_WING;
        int endPart = portion.size - BLUR_VECTOR_WING - 1;
        int count = BLUR_VECTOR_WING + 1;
        double* sourcePixel = portion.source;
        double* destPixelBegin = portion.dest;
        int blurVectorBegin = BLUR_VECTOR_WING;
        for (int x = 0; x < portion.size; ++x) {
            int blurElement = blurVectorBegin;
            double* destPixel = destPixelBegin;
            for (int k = 0; k < count; ++k) {
                *destPixel += *sourcePixel * blurVector[blurElement];
                blurElement++;
                destPixel += portion.step;
            }
            if (beginPart > 0) {
                beginPart--;
                blurVectorBegin--;
                count++;
            } else {
                destPixelBegin += portion.step;
            }
            if (endPart > 0) {
                endPart--;
            } else {
                count--;
            }
            sourcePixel += portion.step;
        }
    }
}

void runBlurThreads(int threadCount)
{
    std::vector<std::thread> threads;
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back(createBlurPortion);
    }
    for (auto& t : threads) {
        t.join();
    }
}

void createBlurMat(cv::Mat& grayImage, cv::Mat& blurImage, int threadCount)
{
    fillBlurVector();
    cv::Mat rowBlurImage(grayImage.rows, grayImage.cols, CV_64FC1, cv::Scalar::all(0.0));
    blurImage.create(grayImage.rows, grayImage.cols, CV_64FC1);
    blurImage = cv::Scalar::all(0.0);

    for (int i = 0; i < grayImage.rows; ++i) {
        blurPortions.push({
            (double*)grayImage.ptr(i, 0),
            (double*)rowBlurImage.ptr(i, 0),
            grayImage.cols,
            1
        });
    }
    runBlurThreads(threadCount);
    
    for (int i = 0; i < rowBlurImage.cols; ++i) {
        blurPortions.push({
            (double*)rowBlurImage.ptr(0, i),
            (double*)blurImage.ptr(0, i),
            rowBlurImage.rows,
            rowBlurImage.cols
        });
    }
    runBlurThreads(threadCount);
}

/*
 ******************************************************************************
 * Other functions
 ******************************************************************************
*/

void convertMatDoubleToInt(cv::Mat& doubleMat, cv::Mat& intMat)
{
    intMat.create(doubleMat.rows, doubleMat.cols, CV_8U);
    double* doubleData = (double*)doubleMat.data;
    uint8_t* intData = intMat.data;
    for (int i = 0; i < doubleMat.rows * doubleMat.cols; ++i) {
        *intData = *doubleData;
        intData++;
        doubleData++;
    }
}

void printUsage()
{
    std::cout << "Usage: conv FILE [THREADS]" << std::endl;
    std::cout << "    FILE    - filename for converting" << std::endl;
    std::cout << "    THREADS - number of threads using for converting" << std::endl;
    std::cout << "              (number of cores of processor by default)" << std::endl;
}

/*
 ******************************************************************************
 * Main function
 ******************************************************************************
*/

int main(int argc, char** argv)
{
    if (argc > 3) {
        printUsage();
        return -1;
    }

    int threadCount;
    if (argc == 3) {
        threadCount = atoi(argv[2]);
    } else {
        threadCount = std::thread::hardware_concurrency();
    }
    if (threadCount <= 0) {
        printUsage();
        return -1;
    }

    if (argc < 2) {
        printUsage();
        return -1;
    }
    cv::Mat loadImage;
    loadImage = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!loadImage.data) {
        std::cout <<  "Could not find or open the image " << argv[1] << "." << std::endl ;
        return -1;
    }

    const auto tic(std::chrono::steady_clock::now());
        
    cv::Mat grayImage;
    createGrayMat(loadImage, grayImage, threadCount);
    
    cv::Mat blurImage;
    createBlurMat(grayImage, blurImage, threadCount);
    
    cv::Mat resultImage;
    convertMatDoubleToInt(blurImage, resultImage);
    
    const auto toc(std::chrono::steady_clock::now());
    std::cout << "Converting time: " << std::chrono::duration<double>(toc - tic).count() * 1000 << " milliseconds." << std::endl;
    
    cv::imwrite("result.jpg", resultImage);
    
    return 0;
}
