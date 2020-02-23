#include <random>
#include <chrono>
#include <type_traits>
#include <sstream>
#include <iostream>

// Not mine, borrowed from stack overflow
template<class Resolution = std::chrono::nanoseconds>
class ExecutionTimer {
public:
    using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                     std::chrono::high_resolution_clock,
                                     std::chrono::steady_clock>;
private:
    const Clock::time_point mStart = Clock::now();

public:
    ExecutionTimer() = default;
    ~ExecutionTimer() {}

    inline int stop() {
        const auto end = Clock::now();
        return std::chrono::duration_cast<Resolution>(end - mStart).count();
    }
};

//
// int popcto_test1(unsigned int bitmap[], int idx){
//     int i = 0,      // index
//         count = 0;  // number of set bits
//     do {
//         // Each node contains 8 bitmaps
//         if(bitmap[i/32] & 1 << (i & 31)){
//             ++count;
//         }
//         ++i;
//     } while (i < idx);
//
//     return count;
// }

// int popcto_test1(unsigned int bitmap[], int idx){
//     const int bmi = idx / 5;        // max bitmap index
//     unsigned int count = 0;         // number of set bits
//     unsigned int temp = 0;
//
//     for (int i = 0; i < bmi; ++i){
//         temp = bitmap[i];
//         temp = temp - ((temp >> 1) & 0b01010101010101010101010101010101);
//         temp = (temp & 0b00110011001100110011001100110011) + ((temp >> 2) & 0b00110011001100110011001100110011);
//         count += (((temp + (temp >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
//     }
//
//     temp = bitmap[bmi] & ((1<<idx)-1);
//     temp = temp - ((temp >> 1) & 0b01010101010101010101010101010101);
//     temp = (temp & 0b00110011001100110011001100110011) + ((temp >> 2) & 0b00110011001100110011001100110011);
//     count += (((temp + (temp >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
//     return count;
// }

#pragma GCC push_options
#pragma GCC optimize ("O0")
// This one serves as an integrity check
int popcto_test2(unsigned int bitmap[], int idx){
    int i = 0,      // index
        count = 0;  // number of set bits
    do {
        // Each node contains 8 bitmaps
        if(bitmap[i/32] & 1 << (i % 32)){
            count++;
        }
        i++;
    } while (i < idx);
    return count;
}
#pragma GCC pop_options


unsigned get_lsb_idx(unsigned v){
    const int debruijn32precomputed[32] = {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
    };
    return debruijn32precomputed[((v & -v)*0x077CB531UL) >> 27];
}

int popcto_test1(unsigned int bitmap[], int idx){
    int i = get_lsb_idx(idx);     // index
    int count = 0;  // number of set bits
    do {
        // Each node contains 8 bitmaps
        if(bitmap[i >> 5] & 1 << (i & 31)){
            ++count;
        }
        ++i;
    } while (i < idx);
    return count;
}

int popcto_test3(unsigned int bitmap[], int idx){
    int count = 0;  // number of set bits
    const int map = idx >> 5;
    for (int i = 0; i < map; ++i){
        // Each node contains 8 bitmaps
        count += __builtin_popcount(bitmap[i]);
    }

    count += __builtin_popcount(bitmap[map] & ((1<<idx)-1));
    return count;
}


int popcto_test4(unsigned int bitmap[], int idx){
    int i = 0,      // index
        j = 0,
        count = 0,  // number of set bits
        map = idx >> 5;
    unsigned int temp = 0;

    while (i < map){
        temp = bitmap[i];
        j = 0;
        while(temp){
            temp &= temp - 1;
            ++j;
        }
        count += j;
        ++i;
    }
    temp = bitmap[i] & ((1<<idx)-1);
    j = 0;
    while(temp){
        temp &= temp - 1;
        ++j;
    }
    return count + j;
}

int popcount(unsigned int temp){
    // This is the fastest popcount for 32-bit types as detailed in "Software Optimization Guide for AMD64 Processors"(179-180)
    temp = temp - ((temp >> 1) & 0b01010101010101010101010101010101);
    temp = (temp & 0b00110011001100110011001100110011) + ((temp >> 2) & 0b00110011001100110011001100110011);
    return (((temp + (temp >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
}

__attribute__((optimize("s")))
int popcto_test6(unsigned int bitmap[], int idx){

    const int bmi = idx >> 5;       // max bitmap index
          int count = 0;            // number of set bits
    //unsigned int _byte4 = 0;

    for (int i = 0; i < bmi; ++i){

        count += popcount(bitmap[i]);
    }
    // This uses the same popcount as above but masks to the required index
    return count + popcount(bitmap[bmi] & ((1<<idx)-1));
}

__attribute__((optimize("s")))
int popcto_test5(unsigned int bitmap[], unsigned idx) {

    const int bmi = idx >> 5;       // max bitmap index
    unsigned  count = 0;            // number of set bits
    unsigned _byte4 = 0;

    for (int i = 0; i < bmi; ++i){
        // This is the fastest popcount for 32-bit types as detailed in "Software Optimization Guide for AMD64 Processors"(179-180)
        _byte4 = bitmap[i];
        _byte4 = _byte4 - ((_byte4 >> 1) & 0b01010101010101010101010101010101);
        _byte4 = (_byte4 & 0b00110011001100110011001100110011) + ((_byte4 >> 2) & 0b00110011001100110011001100110011);
        count += (((_byte4 + (_byte4 >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
    }
    // This uses the same popcount as above but masks to the required index
    _byte4 = bitmap[bmi] & ((1<<idx)-1);
    _byte4 = _byte4 - ((_byte4 >> 1) & 0b01010101010101010101010101010101);
    _byte4 = (_byte4 & 0b00110011001100110011001100110011) + ((_byte4 >> 2) & 0b00110011001100110011001100110011);
    count += (((_byte4 + (_byte4 >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
    return count;
}


int main(){
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(1, ~(1 << 31));

    unsigned int bitmaps[8];
    for (int i = 0; i < 8; ++i){
        bitmaps[i] = static_cast<unsigned int>(dist(mt));
    }

    unsigned long long test1times = 0,
                       test2times = 0,
                       test3times = 0,
                       test4times = 0,
                       test6times = 0,
                       test5times = 0,
                       basetime = 0;

    int result1 = 0,
        result2 = 0,
        result3 = 0,
        result4 = 0,
        result6 = 0,
        result5 = 0;

    long iterations = 150000;

    std::cout << "~Population Count Tests: running " << iterations << " times to get average ns~" << std::endl;

    int example = 193;


    for (int j = 0; j < iterations; j++){
        ExecutionTimer<std::chrono::nanoseconds> baseline;
        basetime += baseline.stop();

        ExecutionTimer<std::chrono::nanoseconds> timer1;
        result1 = popcto_test1(bitmaps, example);
        test1times += (timer1.stop());

        ExecutionTimer<std::chrono::nanoseconds> timer2;
        result2 = popcto_test2(bitmaps, example);
        test2times += (timer2.stop());

        ExecutionTimer<std::chrono::nanoseconds> timer3;
        result3 = popcto_test3(bitmaps, example);
        test3times += (timer3.stop());

        ExecutionTimer<std::chrono::nanoseconds> timer4;
        result4 = popcto_test4(bitmaps, example);
        test4times += (timer4.stop());

        ExecutionTimer<std::chrono::nanoseconds> timer5;
        result5 = popcto_test5(bitmaps, example);
        test5times += (timer5.stop());

        ExecutionTimer<std::chrono::nanoseconds> timer6;
        result6 = popcto_test6(bitmaps, example);
        test6times += (timer6.stop());
    }

    std::cout << "Population Count | Optimization Type         | Time (nanoseconds)" << std::endl
              << "Actual      = "<< result2 << " | No Optimizations(Actual)  | "<< (test2times/iterations) - (basetime/iterations) << std::endl
              << "Test Case 1 = "<< result1 << " | Bitshift Magic Only  (1)  | "<< (test1times/iterations) - (basetime/iterations)<< std::endl
              << "Test Case 3 = "<< result3 << " | __builtin_popcount(3)     | "<< (test3times/iterations) - (basetime/iterations)<< std::endl
              << "Test Case 4 = "<< result4 << " | Peter Wegner -Modified(4) | "<< (test4times/iterations) - (basetime/iterations)<< std::endl
              << "Test Case 5 = "<< result5 << " | Div&Conq Messy Magic(5)   | "<< (test5times/iterations) - (basetime/iterations)<< std::endl
              << "Test Case 6 = "<< result6 << " | Div&Conq Somewhat Clean(6)| "<< (test6times/iterations) - (basetime/iterations)<< std::endl;

    return 0;

    // Test with flags "-mpopcnt", "-funroll-loops", "-O3"
}
