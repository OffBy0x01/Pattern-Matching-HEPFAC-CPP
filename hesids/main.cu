#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <type_traits>
#include <string>
#include <bitset>
#include <algorithm>
#include <vector>
#include <functional>
// In case bitmap count needs to be changed in future - other areas will need changed too but this simplifies part of the issue
#define COUNT_BITMAP    8 // number of bitmaps in each node
#define OFFSET_IDX      8 // Index of offset in texture (usually same as COUNT_BITMAP)
#define TEXTURE_WIDTH   9 // texture width is always bitmaps + 1

// OPTIONAL DEFINES (COMMENT OUT TO DISABLE)
// provides basic information if enabled. (NO COUT if disabled)
#define BASICOUT 1

// provides EXTENSIVE debug information if enabled
//#define VERBOSEOUT 1

// enables bitwise optimizations
#define OPTIMIZATION_BITWISE 1

// enables population count optimizations
#define OPTIMIZATION_POPC 1

// enables output reduction optimizations
#define OPTIMIZATION_OUTPUT 1

// enables async for values greater than 1 - recommended = 8
#define COUNT_STREAMS 16
// Easy toggle;
#ifndef COUNT_STREAMS
    #define COUNT_STREAMS 1
#endif

// enables read-only memory
#define READ_ONLY_CONST 1

// Cuda error handling for non-kernel operations
#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess){printf("\nCUDA Error: %s (Error Number = %d) on line %d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); exit(1);} }

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


// Better device properties structure
struct DeviceProperties : cudaDeviceProp {
    int driverVersion = -1;
    int runtimeVersion = -1;
    int memory_total = -1;
    int memory_available = -1;
};

// Allows easy tracking of node state
using NodeState = enum NodeState {
    root = 0, unset = -1, end = -2, ignore = -3
};

// Raw Node
using NodeReduced = struct NodeReduced{
    unsigned bitmap[8];
    int offset = 0;
};

// 'Full' Node
using Node = struct Node{
    unsigned bitmap[8];
    int offset      = 0;
    int final       = 0;
    int ascii       = 0; // trying to go without it only complicates things when building reduction trie
    int prev_node   = 0;
    int node_id     = 0;
};

// 'Wrapper' for passing tries
using Wrapper = struct Wrapper {
    NodeReduced * nodes;
    int size; // size of trie
};

// 'Pattern' that tracks most recent node related to itself
using Pattern = struct Pattern {
    std::string pattern_str;
    int last_pattern_node_id = 0;

    Pattern(const std::string m_pattern)
        : pattern_str(m_pattern), last_pattern_node_id(0){}

    Pattern(const std::string m_pattern, const int m_last_pattern_node_id)
        : pattern_str(m_pattern), last_pattern_node_id(m_last_pattern_node_id){}

    // only used by reduction trie but saves copying between structures
    int parent_save = NodeState::unset;
    int parent = 0;
    int child = -1;
};

using PacketIdentifier = struct PacketIdentifier {
    unsigned long long id = 0;
    unsigned int begins_at = 0;
    unsigned int ends_at = 0;
};


#ifdef OPTIMIZATION_BITWISE
    #define MOD32(x)    ((x)&31)
    #define MOD8(x)     ((x)&7)
    #define DIV32(x)    ((x)>>5)
    #define DIV8(x)     ((x)>>3)
    #define MUL8(x)     ((x)<<3)
    #define ADD1(x)     ++x
    #define SUB1(x)     --x

#else
    #define MOD32(x)    ((x)%32)
    #define MOD8(x)     ((x)%8)
    #define DIV32(x)    ((x)/32)
    #define DIV8(x)     ((x)/8)
    #define MUL8(x)     ((x)*8)
    #define ADD1(x)     x++
    #define SUB1(x)     x--
#endif

#if defined(OPTIMIZATION_POPC)
    __device__ int
    popcount_texture(unsigned temp){ // This one exists as proof that it works, just that CUDA is an arsehole rn.
        // // This is the fastest popcount for 32-bit types as detailed in "Software Optimization Guide for AMD64 Processors"(179-180)
        // temp = temp - ((temp >> 1) & 0b01010101010101010101010101010101);
        // temp = (temp & 0b00110011001100110011001100110011) + ((temp >> 2) & 0b00110011001100110011001100110011);
        // return (((temp + (temp >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;

        return __popc(temp);
        // int j = 0;
        // while(temp != 0){
        //     temp = temp & (temp - 1);
        //     ++j;
        // }
        //
        // return j;
    }
#else
    __device__ int
    popcount_texture(const unsigned temp){ // This one exists as proof that it works, just that CUDA is an arsehole rn.
        int i = 0,      // index
            count = 0;  // number of set bits
        do {
            if((temp & (1 << i)) != 0){
                // corresponding alphabet section is populated and the index offset is not 0
                ADD1(count);
            }
            ADD1(i);
        } while (i < 32);
        return count;
    }
#endif




__global__ void
#ifdef READ_ONLY_CONST
    search_trie_texture(const cudaTextureObject_t texture_trie, const char * __restrict__ input_text, const int size_text, const int text_offset, unsigned * out) {
#else
    search_trie_texture(cudaTextureObject_t texture_trie, char * input_text, int size_text, int text_offset, unsigned * out) {
#endif
    // Get ID of current thread
    const int tid = blockIdx.x * blockDim.x + threadIdx.x + text_offset;

    // Trie
    int popcount = 0;
    int node_idx = 0;
    int ascii = input_text[tid];
    int ascii_idx = DIV32(ascii);
    int ascii_off = MOD32(ascii);

    // Actual Search...
    for (int i=tid+1; i < size_text && ((tex1Dfetch<int>(texture_trie, node_idx + ascii_idx) & (1 << ascii_off)) != 0); ADD1(i)){
        // get current node by adding popcount to location of first child
        for (int j=0; j < ascii_idx; ADD1(j)){
            popcount += popcount_texture(tex1Dfetch<int>(texture_trie, node_idx + j));
        }
        popcount += popcount_texture(tex1Dfetch<int>(texture_trie, node_idx + ascii_idx) & ((1<<ascii_off)-1));
        // Node_id is where it would normally be, but no structures, so must multiply by elements per node(9) to get true index
        node_idx = TEXTURE_WIDTH*(tex1Dfetch<int>(texture_trie, node_idx + OFFSET_IDX) + popcount);

        popcount = 0;
        // Get current ascii from texture memory for compare
        ascii = input_text[i];
        ascii_idx = DIV32(ascii);
        ascii_off = MOD32(ascii);
    }

    // TODO USE SIGNED (int vs unsigned int)
    #ifdef OPTIMIZATION_OUTPUT
        atomicOr( &out[DIV32(tid)], (tex1Dfetch<int>(texture_trie, node_idx + OFFSET_IDX) == NodeState::end) << (MOD32(tid)) );
    #else
        out[tid] = (tex1Dfetch<int>(texture_trie, node_idx + OFFSET_IDX) == NodeState::end);
    #endif

}

// GPU SEARCH FUNCTION - shared memory
__global__ void
#ifdef READ_ONLY_CONST
    search_trie_shared(NodeReduced * __restrict__ trie_array, const unsigned size_trie, const char * __restrict__ input_text, const unsigned size_text, const int text_offset, unsigned * out) {
#else
    search_trie_shared(NodeReduced * trie_array, unsigned size_trie, char * input_text, unsigned size_text, int text_offset, unsigned * out) {
#endif
    extern __shared__ NodeReduced trie_shared[];

    // Get ID of current thread
    const int tid = blockDim.x * blockIdx.x + threadIdx.x + text_offset;

    // Copy trie to shared memory from global memory
    if (threadIdx.x < warpSize) {
        for(int i = threadIdx.x; i  <size_trie; i += warpSize) {
            trie_shared[i] = trie_array[i];
        }
    }
    __syncthreads();

    // Trie
    unsigned popcount = 0;
    // Current node for search
    NodeReduced * current_node = &trie_shared[0];
    // Pattern and starting character in pattern to compare
    int ascii = input_text[tid];
    int ascii_idx = DIV32(ascii);
    int ascii_off = MOD32(ascii);
    // Actual Search...
    for (int i=tid+1; i < size_text && ((current_node->bitmap[ascii_idx] & (1 << (ascii_off) )) != 0); ADD1(i)){
        // get current node by adding popcount to location of first child
        for (int j=0; j < ascii_idx; ADD1(j)){
            popcount += popcount_texture(current_node->bitmap[j]);
        }
        popcount += popcount_texture(current_node->bitmap[ascii_idx] & ((1<<ascii_off)-1));

        current_node = &trie_shared[(current_node->offset) + popcount];

        popcount = 0;
        // Get current ascii for compare
        ascii = input_text[i];
        ascii_idx = DIV32(ascii);
        ascii_off = MOD32(ascii);
    }

    #ifdef OPTIMIZATION_OUTPUT
        atomicOr( &out[tid/32], (current_node->offset == NodeState::end) << (MOD32(tid)) );
    #else
        out[tid] = (current_node->offset == NodeState::end);
    #endif

}

// GPU SEARCH FUNCTION - global memory
__global__ void
#ifdef READ_ONLY_CONST
    search_trie_global(NodeReduced * __restrict__ trie_array, const char * __restrict__ input_text, const unsigned size_text, const int text_offset, unsigned * out) {
#else
    search_trie_global(NodeReduced * trie_array, char * input_text, unsigned size_text, int text_offset, unsigned * out) {
#endif
    // Get ID of current thread
    const int tid = blockIdx.x * blockDim.x + threadIdx.x + text_offset;

    // Trie
    unsigned popcount = 0;
    // Current node for search
    NodeReduced * current_node = &trie_array[0];
    // Pattern and starting character in pattern to compare
    int ascii = input_text[tid];
    int ascii_idx = DIV32(ascii);
    int ascii_off = MOD32(ascii);

    // Actual Search...
    for (int i=tid+1; i < size_text && ((current_node->bitmap[ascii_idx] & (1 << (ascii_off) )) != 0); ADD1(i)){
        // get current node by adding popcount to location of first child
        for (int j=0; j < ascii_idx; ADD1(j)){
            popcount += popcount_texture(current_node->bitmap[j]);
        }
        popcount += popcount_texture(current_node->bitmap[ascii_idx] & ((1<<ascii_off)-1));

        current_node = &trie_array[(current_node->offset) + popcount];

        popcount = 0;
        ascii = input_text[i];
        ascii_idx = DIV32(ascii);
        ascii_off = MOD32(ascii);
    }

    #ifdef OPTIMIZATION_OUTPUT
        atomicOr( &out[tid/32], (current_node->offset == NodeState::end) << (MOD32(tid)) );
    #else
        out[tid] = (current_node->offset == NodeState::end);
    #endif
}

class Hepfac{
public:
    // Constructors
    Hepfac(unsigned);

    // Destructor
    ~Hepfac();

    // Build/Reduce Helpers
    int popcount_node(Node *, int);
    int popcount_node(NodeReduced *, int);

    // Demo, debug, etc
    bool print_status(const std::string &, bool);
    void print_progress(const std::string &);
    void print_info(const std::string &);
    void print_config();
    void set_test_loops(int);

    // Must call one of these build from methods
    void build_from_file(std::string&);
    void build_from_string_vector(std::vector<std::string> &);

    // Build trie
    void init_trie();
    bool build_trie();
    bool build_reduced_trie();

    // Verify Trie
    bool verify_trie();
    bool verify_reduced_trie();
    bool verify_texture_trie(int*);


    // set source as file <filename>
    void set_source_file(std::string &);
    // set source as string
    void set_source_string(std::string &);

    // Search Trie
    int search_global();
    int search_shared();
    int search_texture();
    int search_bench();

    // Matches
    void set_match_callback(std::function<void(int)>);
    void set_bit_index_size(int);
    int get_matches(unsigned*);
    int get_lsb_idx(unsigned);

private:
    // Patterns
    int pattern_count = 0;
    int min_signature_len = 6;
    int max_signature_len = 12;
    std::vector<Pattern> pattern_vector = {};
    // Trie build
    std::vector<Node> trie;
    int node_count = -1;
    // Trie Reduction
    NodeReduced * trie_reduced = nullptr;
    int node_reduced_count = -1;
    // Search Source
    std::string *current_buffer = nullptr;
    std::string file_buffer;

    // Device parameters
    const int threads_per_block = 128;
    const int count_streams = COUNT_STREAMS;

    // Number of test loops to run
    int tests = 25;

    // Data Size
    size_t text_size = 0;
    size_t trie_size = 0;
    size_t outp_size = 0;
    // Device Properties
    int device_count = -1;
    std::vector<DeviceProperties> device_properties;
    // Event Timers
    cudaEvent_t start;
    cudaEvent_t stop;

    // Results
    const int debruijn32precomputed[32] = {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
    };
    std::function<void(int)> match_callback;
    int result_size_bits = 0;               // number of bits allocated (elements*8*sizeof(type))
    int result_elements = 0;                // Number of elements allocated
    int result_size_elements = 0;           // Size of elements allocated (elements*sizeof(type)) - for memcpy
    int result_size_element = sizeof(int);  // Size of individual element
};

// Construct with max signature length
Hepfac::Hepfac(unsigned m_max){
    std::cout << "\033[91m[!] New Hepfac Instance \033[0m" << std::endl;
    print_config();
    max_signature_len = m_max;
    // CUDA SETUP
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

// Destructor
Hepfac::~Hepfac(){
    // Deallocate Events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    try {
        if (trie_reduced != nullptr){
            // might have already been deleted - don't want to crash on exit
            delete trie_reduced;
        }
    } catch (std::exception e){
        // TODO ^ don't use this generic shit
    }
}

void Hepfac::set_test_loops(int loops){
    tests = loops;
}

// Count bits set in a given reduced node
int Hepfac::popcount_node(NodeReduced* current_node, int idx){
    const int bmi = idx >> 5;        // max bitmap index
    unsigned  count = 0;             // number of set bits
    unsigned div_and_conq = 0;

    for (int i = 0; i < bmi; ++i){
        // This is the fastest popcount for 32-bit types as detailed in "Software Optimization Guide for AMD64 Processors"(179-180)
        div_and_conq = current_node->bitmap[i];
        div_and_conq = div_and_conq - ((div_and_conq >> 1) & 0b01010101010101010101010101010101);
        div_and_conq = (div_and_conq & 0b00110011001100110011001100110011) + ((div_and_conq >> 2) & 0b00110011001100110011001100110011);
        count += (((div_and_conq + (div_and_conq >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
    }
    // This uses the same popcount as above but masks to the required index
    div_and_conq = current_node->bitmap[bmi] & ((1<<idx)-1);
    div_and_conq = div_and_conq - ((div_and_conq >> 1) & 0b01010101010101010101010101010101);
    div_and_conq = (div_and_conq & 0b00110011001100110011001100110011) + ((div_and_conq >> 2) & 0b00110011001100110011001100110011);
    count += (((div_and_conq + (div_and_conq >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
    return count;
}

// Count bits set in a given 'full' node
int Hepfac::popcount_node(Node* current_node, int idx){
    const int bmi = idx >> 5;        // max bitmap index
    unsigned  count = 0;             // number of set bits
    unsigned div_and_conq = 0;

    for (int i = 0; i < bmi; ++i){
        // This is the fastest popcount for 32-bit types as detailed in "Software Optimization Guide for AMD64 Processors"(179-180)
        div_and_conq = current_node->bitmap[i];
        div_and_conq = div_and_conq - ((div_and_conq >> 1) & 0b01010101010101010101010101010101);
        div_and_conq = (div_and_conq & 0b00110011001100110011001100110011) + ((div_and_conq >> 2) & 0b00110011001100110011001100110011);
        count += (((div_and_conq + (div_and_conq >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
    }
    // This uses the same popcount as above but masks to the required index
    div_and_conq = current_node->bitmap[bmi] & ((1<<idx)-1);
    div_and_conq = div_and_conq - ((div_and_conq >> 1) & 0b01010101010101010101010101010101);
    div_and_conq = (div_and_conq & 0b00110011001100110011001100110011) + ((div_and_conq >> 2) & 0b00110011001100110011001100110011);
    count += (((div_and_conq + (div_and_conq >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
    return count;
}

// use to provide helpful 'progress' banners
void Hepfac::print_progress(const std::string& message){
    #if defined(VERBOSEOUT) || defined(BASICOUT)
        std::string padding(message.length(), '-');
        std::cout   << "\033[94m" << padding << "\033[0m" << std::endl
                    << "\033[96m[#] " << message << "\033[0m" <<std::endl
                    << "\033[94m" << padding << "\033[0m" << std::endl;
    #endif
}

bool Hepfac::print_status(const std::string& message, bool condition){
    #if defined(VERBOSEOUT) || defined(BASICOUT)
        if (condition){
            std::cout << "\033[92m[+] " << message << " PASS\033[0m" <<std::endl;
        } else {
            std::cout << "\033[91m[-] " << message << " FAIL\033[0m" <<std::endl;
        }
    #endif
    return condition;
}

void Hepfac::print_info(const std::string& message){
    #if defined(VERBOSEOUT)  || defined(BASICOUT)
        std::cout << "\033[94m[!] " << message << "\033[0m" <<std::endl;
    #endif
}

void Hepfac::print_config(){
    // provides EXTENSIVE debug information if enabled
    #if defined(VERBOSEOUT)
        std::cout <<"\033[93m[$] " << "DEBUG LEVEL 'VERBOSE'" << "\033[0m" <<std::endl;
    // provides BASIC debug information if enabled.
    #elif defined(BASICOUT)
        std::cout <<"\033[93m[$] " << "DEBUG LEVEL 'BASIC'" << "\033[0m" <<std::endl;
    #endif

    // enables bitwise optimizations
    #if defined(OPTIMIZATION_BITWISE)
        std::cout <<"\033[93m[$] " << "OPTIMIZATION_BITWISE ENABLED" << "\033[0m" <<std::endl;
    #endif

    // enables population count optimizations
    #if defined(OPTIMIZATION_POPC)
        std::cout <<"\033[93m[$] " << "OPTIMIZATION_POPC ENABLED" << "\033[0m" <<std::endl;
    #endif

    // enables output reduction optimizations
    #if defined(OPTIMIZATION_OUTPUT)
        std::cout <<"\033[93m[$] " << "OPTIMIZATION_OUTPUT ENABLED" << "\033[0m" <<std::endl;
    #endif

    #if COUNT_STREAMS != 1
        std::cout <<"\033[93m[$] " << "ASYNC ENABLED (COUNT_STREAMS="<< COUNT_STREAMS <<")" << "\033[0m" <<std::endl;
    #endif

    #if defined(READ_ONLY_CONST)
        std::cout <<"\033[93m[$] " << "READ_ONLY_CONST ENABLED" << "\033[0m" <<std::endl;
    #endif


}
// ------------------
// Build From Methods
// ------------------

// Build From File
void Hepfac::build_from_file(std::string &filename){
    print_progress("Building from file");
    std::ifstream file(filename, std::ifstream::in);
    std::string line;

    if (!file.is_open()){
        std::cout << "\033[91m" << "[-] File not found FAIL"  << "\033[0m" <<std::endl;
        throw("File not found");
    }

    while(getline(file, line)){
        if (line.size() < min_signature_len){
            // might throw on file with empty last line
            std::cout << "\033[91m" << "[-] Signature below reduction size FAIL"  << "\033[0m" <<std::endl;
            throw("Signature less than recommended minimum");
        }
        pattern_vector.emplace_back(Pattern(line));
        pattern_count++;
    }

    std::cout << "\033[92m" << "[+] File read successfully PASS" << "\033[0m" <<std::endl;

    init_trie();
    build_trie();
    build_reduced_trie();
}

// Build From String Vector
void Hepfac::build_from_string_vector(std::vector<std::string> & string_vector){
        print_progress("Building from string vector");
    for (auto str : string_vector){
        pattern_vector.emplace_back(Pattern(str));
    }
    pattern_count = string_vector.size();

    init_trie();
    build_trie();
    build_reduced_trie();
}

// ----------------
// helper Functions
// ----------------

// std::string query_device(bool print){
//     // Reset & Update Device Count
//     device_count = 0;
//     cudaGetDeviceCount(&device_count);
//     device_properties.resize(device_count);
//
//     // Generate Output String
//     std::ostringstream output;
//     output << device_count << " CUDA Capable device(s) detected" << std::endl;
//
//     for (int dev = 0; dev < device_count; ++dev){
//         cudaSetDevice(dev);
//         cudaGetDeviceProperties(&device_properties[dev], dev);
//         cudaDriverGetVersion(&device_properties[dev].driverVersion);
//         cudaRuntimeGetVersion(&device_properties[dev].runtimeVersion);
//         cudaMemGetInfo(&device_properties[dev].memory_available, &device_properties[dev].memory_total);
//         // Generate Individual Device Output String
//         output  << "Device " << dev << "(" << device_properties[dev].name << "): " << std::endl
//                 << "    'CUDA Driver Version'=" << device_properties[dev].driverVersion / 1000<<"." <<(device_properties[dev].driverVersion % 100) / 10 << std::endl
//                 << "    'Runtime Version'= " << device_properties[dev].runtimeVersion / 1000 << "." << (device_properties[dev].runtimeVersion % 100) / 10 << std::endl
//                 << "    'CUDA Capability Major/Minor version number'=" << device_properties[dev].major << "/" << device_properties[dev].minor << std::endl
//                 << "    'GPU Max Clock rate'=" << device_properties[dev].clockRate << std::endl
//                 << "    'Memory (Free)'=" << device_properties[dev].memory_available << std::endl
//                 << "    'Memory (Total)'=" << device_properties[dev].memory_total << std::endl << std::endl;
//     }
//
//     if (print){
//         std::cout << output.str();
//     }
//
//     return output.str();
// }

// -----------------
// Trie Construction
// -----------------

void Hepfac::init_trie(){
    print_progress("Initializing trie");

    // Patterns must be sorted before building trie
    std::sort(pattern_vector.begin(), pattern_vector.end(),[](const Pattern &p1, const Pattern &p2 ) -> bool {
        // TODO might need to make this < 0;
        return p1.pattern_str < p2.pattern_str;
    });
    std::cout << "\033[94m" << "[!] Sorted " << pattern_vector.size()  << " signatures" << "\033[0m" <<std::endl;


    trie.resize(pattern_count * max_signature_len);
    for(auto node : trie){
        for(int bitmap_idx=0; bitmap_idx < COUNT_BITMAP ;bitmap_idx++){
            node.bitmap[bitmap_idx]=0;
        }
        node.prev_node = 0;
        node.offset = 0;
        node.final = 0;
    }

    std::cout << "\033[94m" << "[!] Initialized nodes to 0" << "\033[0m" <<std::endl;

}

// Build trie
bool Hepfac::build_trie(){
    print_progress("Building trie");

    #if defined(VERBOSEOUT) || defined(BASICOUT)
        ExecutionTimer<std::chrono::nanoseconds> exec_timer;
    #endif

    int ascii = -1;
    int node_id = 0; // Used as both Node ID and Position of Node in Array
    Node * current_node = &trie[0];   // Start at Root
    Pattern * current_pattern = nullptr;

    // character in pattern
    for (int pattern_depth = 0; pattern_depth < max_signature_len; pattern_depth++){
        // pattern in pattern list
        for (int current_pattern_pos = 0; current_pattern_pos < pattern_count && pattern_vector[current_pattern_pos].pattern_str.size() > pattern_depth; current_pattern_pos++){
            // Get current pattern
            current_pattern = &pattern_vector[current_pattern_pos];
            // Get letter at pattern_depth from current pattern
            ascii = static_cast<int>(current_pattern->pattern_str[pattern_depth]);
            // Get most recent related Node (by id) from current pattern
            current_node = &trie[current_pattern->last_pattern_node_id];

            // Has the character already appeared in this (parent) node?
            if ((current_node->bitmap[DIV32(ascii)] & (1 << MOD32(ascii) )) != 0){
                // Yes, it has; *point last_pattern_node_id at that node id
                current_pattern->last_pattern_node_id = popcount_node(current_node, ascii) + current_node->offset;
            }
            else {
                // Nope! Set 'character exists' in this (parent) node
                current_node->bitmap[DIV32(ascii)] |= (1 << MOD32(ascii));
                // If the first child has not already been set for current node
                if (current_node->offset == 0){
                    current_node->offset = node_id + 1; // node number + 1 is what the next created node ID will be
                }
                // 'create' new Node
                trie[++node_id].prev_node = current_node->node_id; // set prev node of child to current node
                current_node = &trie[node_id]; // set current node to child
                // Get new Node ID
                current_node->node_id = node_id;
                current_node->ascii = ascii;
                // Update last related node of current pattern
                current_pattern->last_pattern_node_id = current_node->node_id;
            }
        }
    }
    std::stringstream str;
    long timer = exec_timer.stop();
    str << "Built trie of " << current_node->node_id << " nodes in " << timer <<"ns / " << timer/1000 <<"ms";
    print_info(str.str());
    return verify_trie();
}

// Verify Trie Construction
bool Hepfac::verify_trie(){
    std::cout << "\033[94m" << "[!] Verifying  " << pattern_vector.size() << " patterns" << "\033[0m" <<std::endl;
    int ascii = -1;
    int popcount = 0;
    int matches = 0;
    Node * current_node = nullptr;
    Pattern * current_pattern = nullptr;

    // For patterns
    for (int i = 0; i < pattern_count; i++){
        current_node = &trie[0];
        current_pattern = &pattern_vector[i];
        #if defined(VERBOSEOUT)
            std::cout <<  "Verifying Pattern: " << current_pattern->pattern_str << std::endl;
        #endif
        // For character in each pattern
        for (int j = 0; j < max_signature_len; j++){
            ascii = static_cast<int>(current_pattern->pattern_str[j]);
            popcount = 0;
            // If character present in node
            if ((current_node->bitmap[DIV32(ascii)] & (1 << MOD32(ascii) )) != 0){
                popcount = popcount_node(current_node, ascii);
                #if defined(VERBOSEOUT)
                    // std::cout <<
                    // " '" << current_pattern->pattern_str[j] << "'=present," <<
                    // " 'ID'=" << current_node->node_id << "," <<
                    // " 'popcount to idx'" << popcount << "," <<
                    // " 'offset'=" << current_node->offset << std::endl;
                    std::cout << "[+] Char: " << current_pattern->pattern_str[j] <<std::endl;
                #endif
                current_node = &trie[(current_node->offset) + popcount];

            // If character not present in node
            } else {
                #if defined(VERBOSEOUT)
                    std::cout << "[-] Char: " << current_pattern->pattern_str[j] <<std::endl;
                    std::cout <<
                    " '" << current_pattern->pattern_str[j] << "'=not present," <<
                    " 'ID'=" << current_node->node_id << "," << std::endl;
                    for (int k = 0; k < COUNT_BITMAP; k++){
                        std::cout << std::bitset<COUNT_BITMAP>(current_node->bitmap[i]) << " | ";
        	        }
                    std::cout << std::endl;

                #endif

            }
        } // end of pattern match loop

        if(current_node->ascii == current_pattern->pattern_str[max_signature_len-1]){
            matches++;
            #if defined(VERBOSEOUT)
                std::cout << "\033[92m" << "[+] Signature " << i  << " PASS" << "\033[0m" <<std::endl;
            #endif
            current_node = &trie[(current_node->offset)];
        } else {
            std::cout << "\033[91m" << "[-] Signature " << i  << " FAIL"  << "\033[0m" <<std::endl;
            exit(1);
        }
    } // end of pattern selection loop

    return print_status("Build trie verification", matches == pattern_count);
}

// --------------
// Trie Reduction
// --------------

// Reduce Trie
bool Hepfac::build_reduced_trie(){
    print_progress("[+] Building reduced trie");
    #if defined(VERBOSEOUT) || defined(BASICOUT)
        ExecutionTimer<std::chrono::nanoseconds> exec_timer;
    #endif
    // get last node
    node_count = pattern_vector[pattern_count -1].last_pattern_node_id;
    int current_node_index = node_count - 1;
    // Remove last nodes, copy patterns to reduction patterns
    for (int p = 0; p < pattern_count; p++){
        // inform grandparents of new parent location
        trie[trie[pattern_vector[p].last_pattern_node_id].prev_node].offset = node_count;
        // get new parent nodes
        pattern_vector[p].parent = trie[pattern_vector[p].last_pattern_node_id].prev_node;
        //point children to END
        pattern_vector[p].child = node_count;
        pattern_vector[p].pattern_str = pattern_vector[p].pattern_str;
    }
    // 'Create' END
    trie[node_count].offset = NodeState::end;
    trie[node_count].ascii = NodeState::end;
    trie[node_count].node_id = node_count;
    // Compare current node/pattern with another
    Pattern * current_pattern = nullptr;
    Pattern * compare_pattern = nullptr;
    // Nodes for comparison
    Node * current_node = &trie[node_count];
    Node * compare_node = nullptr;
    Node * new_node = nullptr;
    bool all_children_match = false;

    // will only exit once all patterns reach root
    while (current_node->prev_node != NodeState::root){
        // step through patterns in reverse
        for (int p =  pattern_count -1; p >= 0; p--){
            // get existing node from trie
            current_pattern = &pattern_vector[p];
            current_node = &trie[current_pattern->parent];
            // if this node has been processed already fetch another
            if (NodeState::ignore == current_node->node_id){
                continue;
            }
            // Check for similar nodes
            for (int processed_p = pattern_count - 1; processed_p > p; processed_p--){
                // get node from trie, skip duplicates
                compare_pattern = &pattern_vector[processed_p];
                compare_node = &trie[compare_pattern->parent];
                if (compare_node->node_id == current_node->node_id){
                    continue;
                }
                // Ensure potential match has exact same children & parents do to (requirement for merging without disrupting the structure)
                all_children_match = true;
                for (int i = 0; i < COUNT_BITMAP; i++){
                    if (current_node->bitmap[i] != compare_node->bitmap[i] || trie[current_node->prev_node].bitmap[i] != trie[compare_node->prev_node].bitmap[i]){
                        all_children_match = false;
                        break;
                    }
                }
            }
            // nodes were found that can be merged (found matching set and part of a linkable pattern)
            if(all_children_match && current_node->offset == compare_node->offset){
                if (trie[current_node->prev_node].offset == current_pattern->parent){
                    // inform grandparent where first child is now located
                    trie[current_node->prev_node].offset = compare_node->node_id;
                }
                // Update patterns to account for merge
                current_pattern->parent = current_pattern->parent;

            } else {
                // children aren't shared so cannot be merged (must create new node)

                // Repurpose node and copy data
                new_node = &trie[current_node_index];
                for (int i=0; i < COUNT_BITMAP; i++){
                    new_node->bitmap[i] = trie[current_pattern->parent].bitmap[i];
                }
                new_node->ascii = trie[current_pattern->parent].ascii;
                new_node->offset = trie[current_pattern->parent].offset;
                new_node->prev_node = current_node->prev_node;
                new_node->node_id = current_node_index;

                // Ignore current node in future operations
                current_node->node_id = NodeState::ignore;

                if (trie[current_node->prev_node].offset == current_pattern->parent){
                    // inform grandparent where first child is now located
                    trie[current_node->prev_node].offset = new_node->node_id;
                }
                // Update patterns & index to account for move
                current_pattern->parent = current_node_index--;
            }
        }

        // Move up to next pattern level
        for (auto & pat : pattern_vector){
            pat.parent = trie[pat.parent].prev_node;
        }
    }

    // Copy data from old root to new
    new_node = &trie[current_node_index];
    current_node = &trie[0];

    for (int i=0; i < COUNT_BITMAP; i++){
        new_node->bitmap[i] = current_node->bitmap[i];
    }
    new_node->ascii = NodeState::unset;
    new_node->offset = current_node->offset;
    new_node->node_id = current_node_index;

    // TODO REMOVE AFTER DEBUG
    current_node->node_id = NodeState::ignore;

    // Allocate new trie for size required
    node_reduced_count = node_count + 1 - current_node_index;
    trie_reduced = new NodeReduced[node_reduced_count];

    // Copy trie to TrieReduced
    for (int old_location = current_node_index; old_location < (node_count + 1); old_location++){
        // copy data from offset-location (not 'offset') to actual location
        for (int b = 0; b < COUNT_BITMAP; b++){
            trie_reduced[old_location - current_node_index].bitmap[b] = trie[old_location].bitmap[b];
        }
        if (trie[old_location].offset >= 0){
            trie_reduced[old_location - current_node_index].offset = trie[old_location].offset - current_node_index;
        } else {
            trie_reduced[old_location - current_node_index].offset = trie[old_location].offset;
        }
    }

    std::ostringstream reduced_by;
    long timer = exec_timer.stop();
    reduced_by << "[!] Trie reduced by " << 100 - (100 *(float(node_reduced_count) / float(node_count))) << "%"
               << " (" << node_count<<"->"<<(node_reduced_count) <<") in " << timer << "ns / " << timer/100 << "ms";
    std::cout  << "\033[94m" << reduced_by.str() << "\033[0m" <<std::endl;

    return verify_reduced_trie();
}

// Verify Trie Reduction
bool Hepfac::verify_reduced_trie(){
    std::cout << "\033[94m" << "[!] Verifying  " << pattern_vector.size() << " patterns" << "\033[0m" <<std::endl;
    int ascii = -1;
    int popcount = 0;
    int matches = 0;
    NodeReduced * current_node = nullptr;
    Pattern * current_pattern = nullptr;

    for (int i = 0; i < pattern_count; i++){
        current_node = &trie_reduced[0];
        current_pattern = &pattern_vector[i];
        #if defined(VERBOSEOUT)
            std::cout << "Verifying Pattern: " << current_pattern->pattern_str << std::endl;
        #endif
        for (int j = 0; j < max_signature_len; j++){
            ascii = static_cast<int>(current_pattern->pattern_str[j]);
            popcount = 0;

            if ((current_node->bitmap[ascii/32] & (1 << (ascii & 31) )) != 0){
                #if defined(VERBOSEOUT)
                    std::cout << "[+] Char: " << current_pattern->pattern_str[j] <<std::endl;
                #endif
                popcount = popcount_node(current_node, ascii);
                current_node = &trie_reduced[(current_node->offset) + popcount];
            } else {
                #if defined(VERBOSEOUT)
		            std::cout << "[-] Char: " << current_pattern->pattern_str[j] <<std::endl;
                #endif
                for (int k = 0; k < COUNT_BITMAP; k++){
                    std::cout << std::bitset<COUNT_BITMAP>(current_node->bitmap[i]) << " | ";
		        }
                std::cout << std::endl;
            }
        }
        if(current_node->offset == NodeState::end){
            matches++;
            #if defined(VERBOSEOUT)
                std::cout << "\033[92m" << "[+] Signature " << i  << " PASS" << "\033[0m" <<std::endl;
            #endif
            current_node = &trie_reduced[(current_node->offset)];
        } else {
            std::cout << "\033[91m" << "[-] Signature " << i  << " FAIL"  << "\033[0m" <<std::endl;
            exit(1);
        }
    }
    return print_status("Reduce trie verification", matches == pattern_count);
}


unsigned popc_verify(unsigned temp){
    // This is the fastest popcount for 32-bit types as detailed in "Software Optimization Guide for AMD64 Processors"(179-180)
    temp = temp - ((temp >> 1) & 0b01010101010101010101010101010101);
    temp = (temp & 0b00110011001100110011001100110011) + ((temp >> 2) & 0b00110011001100110011001100110011);
    return (((temp + (temp >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
}

// Verify Trie Reduction
bool Hepfac::verify_texture_trie(int texture_trie[]){
    std::cout << "\033[94m" << "[!] Verifying  " << pattern_vector.size() << " patterns" << "\033[0m" <<std::endl;
    int ascii = -1;
    int popcount = 0;
    int matches = 0;
    int node_idx = 0;
    int ascii_idx = 0;
    const int offset = 8;
    Pattern * current_pattern = nullptr;

    for (int i = 0; i < pattern_count; i++){
        current_pattern = &pattern_vector[i];
        #if defined(VERBOSEOUT)
            std::cout << "Verifying Pattern: " << current_pattern->pattern_str << std::endl;
        #endif
        for (int j = 0; j < max_signature_len; j++){

            ascii = static_cast<int>(current_pattern->pattern_str[j]);
            ascii_idx = DIV32(ascii);

            if ((texture_trie[node_idx + ascii_idx] & (1 << MOD32(ascii) )) != 0){
                #if defined(VERBOSEOUT)
                    std::cout << "[+] Char: " << current_pattern->pattern_str[j] <<std::endl;
                #endif
                for(int k=0; k < ascii_idx; ADD1(k)){
                    popcount += popc_verify(texture_trie[node_idx + k]);
                }
                popcount += popc_verify(texture_trie[node_idx + ascii_idx] & ((1<<MOD32(ascii))-1));

                node_idx = TEXTURE_WIDTH*(texture_trie[node_idx + offset] + popcount);
            } else {
                #if defined(VERBOSEOUT)
		            std::cout << "[-] Char: " << current_pattern->pattern_str[j] <<std::endl;
                #endif
                for (int k = 0; k < COUNT_BITMAP; k++){
                    std::cout << std::bitset<COUNT_BITMAP>(texture_trie[node_idx + k]) << " | ";
		        }
                std::cout << std::endl;
            }
            popcount = 0;
        }
        if(texture_trie[node_idx + offset] == NodeState::end){
            matches++;
            #if defined(VERBOSEOUT)
                std::cout << "\033[92m" << "[+] Signature " << i  << " PASS" << "\033[0m" <<std::endl;
            #endif
            node_idx = 0;
        } else {
            #if defined(VERBOSEOUT)
                std::cout << "\033[91m" << "[-] Signature " << i  << " FAIL"  << "\033[0m" <<std::endl;
            #endif
        }
    }
    return print_status("Texture trie verification", matches == pattern_count);
}

// -------------
// Search Source
// -------------

void Hepfac::set_source_file(std::string & filename){
    print_progress("Getting source (file)");
    // attempt to open file
    std::ifstream file(filename, std::iostream::in);
    if (!file.is_open()){
        throw("Could not open file");
    }

    // update current buffer to point at file string
    if (current_buffer != nullptr){
        delete current_buffer;
    }

    // Allocate Space and copy file to buffer
    std::stringstream buffer;
    buffer << file.rdbuf();
    current_buffer = new std::string(buffer.str());
    file.close();

    // Determine sizes to ease malloc calculations
    text_size = current_buffer->size() * sizeof(char);
    trie_size = node_reduced_count * sizeof(NodeReduced);
    outp_size = current_buffer->size() * sizeof(int);

    std::cout << "\033[94m" << "[!] Opened file of size "<< current_buffer->size() << "B\033[0m" <<std::endl;
}

void Hepfac::set_source_string(std::string & source){
    current_buffer = &source;
    // Determine sizes to ease malloc calculations
    text_size = current_buffer->size() * sizeof(char);
    trie_size = node_reduced_count * sizeof(NodeReduced);
    outp_size = current_buffer->size() * sizeof(int);
}

// -------------------
// HEPFAC Bit Indexing
// -------------------

void Hepfac::set_bit_index_size(int bs){
    result_size_bits = bs;
    result_size_element = sizeof(int)*8;

    #if defined(OPTIMIZATION_OUTPUT)
        result_elements = ((bs / result_size_element) + !!(bs & (result_size_element-1)));
    #else
        result_elements = bs;
    #endif

    result_size_elements = result_elements*sizeof(int);

    #if defined(OPTIMIZATION_OUTPUT)
        outp_size = result_size_elements;
    #endif

    #if defined(VERBOSEOUT) || defined(BASICOUT)
        std::stringstream outsizestr;
        outsizestr << "Allocating output["<<result_elements<<"] ("<< outp_size << "B/"<< outp_size*8 <<"b) ";
        print_info(outsizestr.str());
    #endif
}

void Hepfac::set_match_callback(std::function<void(int)> callback_func){
    match_callback = callback_func;
}

int Hepfac::get_lsb_idx(unsigned v){
    return 0;
    //return (v < 16777216) ? debruijn32precomputed[((v & -v)*0x077CB531UL) >> 27] : 31; // This is often slower
}

// Get all matches
int Hepfac::get_matches(unsigned output[]) {
    print_info("Fetching results");
    #if defined(VERBOSEOUT) || defined(BASICOUT)
        ExecutionTimer<std::chrono::milliseconds> exec_timer;
    #endif
    int matches = 0;
    #if defined(OPTIMIZATION_OUTPUT)
        // must have set a lambda or function reference as callback_func
        for (int i=0; i<result_elements; ++i){
            if(output[i]){
                for(int j=0; j<32; ++j){
                    if (output[i] & (1 << MOD32(j))){
                        ++matches;
                        match_callback((i*32)+j);
                    }
                }
            }
        }
    #else
        for (int i = 0; i < result_elements; ++i){
            if(output[i]){
                ++matches;
                match_callback(i);
            }
        }
    #endif

    #if defined(VERBOSEOUT) || defined(BASICOUT)
        std::stringstream matchstr;
        matchstr << matches << " signature match(es) identified in " << exec_timer.stop() << "ms";
        print_status(matchstr.str(), !!matches);

    #endif

    return matches;
}

// ---------------------
// HEPFAC Search Methods
// ---------------------

// Launch Global-mem-Based Search
int Hepfac::search_global(){
    print_progress("Launching global search");

    // Allocate Host
    set_bit_index_size(current_buffer->length());
    unsigned *output_host;
    char *text_host;

    CUDA_CALL( cudaMallocHost(&text_host, current_buffer->size()) );
    CUDA_CALL( cudaMemcpy(text_host, current_buffer->c_str(), text_size, cudaMemcpyHostToDevice) ); // Text to be searched
    CUDA_CALL( cudaMallocHost(&output_host, outp_size) );
    memset(output_host, 0, result_elements);

    // Allocate Device
    char *text_device;
    NodeReduced *trie_device;
    unsigned *output_device;

    CUDA_CALL( cudaMalloc(&text_device, text_size) );
    CUDA_CALL( cudaMalloc(&trie_device, trie_size) );
    CUDA_CALL( cudaMalloc(&output_device, outp_size) );
    // Trie would only be copied once so not included it in timings
    CUDA_CALL( cudaMemcpy(trie_device, trie_reduced, trie_size, cudaMemcpyHostToDevice) ); // Trie for hepfac


    // Set device parameters
    int blocks_per_grid = (current_buffer->length() + threads_per_block -1) / threads_per_block;
    int text_size_stream = current_buffer->length()/count_streams;
    int output_size_stream = result_elements/count_streams;
    int blocks = (blocks_per_grid/count_streams) ? blocks_per_grid/count_streams : 2;
    // split work across multiple streams
    cudaStream_t streams[count_streams];
    for (int i = 0; i < count_streams; ++i){
        //cudaStreamCreate(&streams[i]);
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    std::stringstream kernel_launch;
    kernel_launch << "Launching "<<tests<<"x search_trie_global<<<" << blocks+1 << ", " << threads_per_block << ",  0, streams["<<count_streams<<"]>>>";
    print_info(kernel_launch.str());

    double total_exec_time = 0;
    for (int i=0; i < tests; ++i){
        cudaEventRecord(start);

        {
            int text_offset = 0;
            int output_offset = 0;

            for (int i = 0; i < count_streams-1; ADD1(i)){
                // No need to copy output thanks to new design.
                CUDA_CALL(cudaMemcpyAsync(&text_device[text_offset], &text_host[text_offset], text_size_stream, cudaMemcpyHostToDevice, streams[i]));
                search_trie_global<<<blocks+1, threads_per_block, 0, streams[i]>>>(trie_device, text_device, current_buffer->size(), text_offset, output_device);
                CUDA_CALL(cudaMemcpyAsync(&output_host[output_offset], &output_device[output_offset], output_size_stream*sizeof(int), cudaMemcpyDeviceToHost, streams[i]));

                output_offset = (i+1)*output_size_stream;
                text_offset = (i+1)*text_size_stream;
            }

            // No need to copy output thanks to new design.
            CUDA_CALL(cudaMemcpyAsync(&text_device[text_offset], &text_host[text_offset], text_size_stream+(current_buffer->length()%count_streams), cudaMemcpyHostToDevice, streams[count_streams-1]));
            search_trie_global<<<blocks+1, threads_per_block, 0, streams[count_streams-1]>>>(trie_device, text_device, current_buffer->size(), text_offset, output_device);
            CUDA_CALL(cudaMemcpyAsync(&output_host[output_offset], &output_device[output_offset], (output_size_stream+(result_elements%count_streams))*sizeof(int), cudaMemcpyDeviceToHost, streams[count_streams-1]));
        }
        // streams are non-blocking, without this throughput will measure (incorrectly) 10k+Gbps
        cudaDeviceSynchronize();
        // Stop event timer
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_exec_time += milliseconds;
    }
    std::stringstream timestr;
    timestr << "Average time to execute " << (total_exec_time/tests) << "ms";
    std::stringstream bandwidthstr;
    bandwidthstr << "Average effective Bandwidth: " << current_buffer->size()*8/(total_exec_time/tests)/1e6 << "Gb/s";
    print_info(timestr.str());
    print_info(bandwidthstr.str());

    // Fetch results
    int matches = get_matches(output_host);

    // Destroy stream and deallocate memory
    for (int i = 0; i < count_streams; ++i){
        cudaStreamDestroy(streams[i]);
    }

    CUDA_CALL( cudaFree(text_device));
    CUDA_CALL( cudaFree(trie_device));
    CUDA_CALL( cudaFree(output_device));
    CUDA_CALL( cudaFreeHost(text_host));
    CUDA_CALL( cudaFreeHost(output_host));

    // TODO
    return 0;
}

// Launch shared-mem-Based Search
int Hepfac::search_shared(){
    print_progress("Launching shared search");
    // Set Shared Preference for this search
    cudaFuncSetCacheConfig(search_trie_shared, cudaFuncCachePreferShared);

    // Allocate Host
    set_bit_index_size(current_buffer->length());
    unsigned *output_host;
    char *text_host;

    CUDA_CALL( cudaMallocHost(&text_host, current_buffer->size()) );
    CUDA_CALL( cudaMemcpy(text_host, current_buffer->c_str(), text_size, cudaMemcpyHostToDevice) ); // Text to be searched
    CUDA_CALL( cudaMallocHost(&output_host, outp_size) );
    memset(output_host, 0, result_elements);

    // Allocate Device
    char *text_device;
    NodeReduced *trie_device;
    unsigned *output_device;

    CUDA_CALL( cudaMalloc(&text_device, text_size) );
    CUDA_CALL( cudaMalloc(&trie_device, trie_size) );
    CUDA_CALL( cudaMalloc(&output_device, outp_size) );
    // Trie would only be copied once so not included it in timings
    CUDA_CALL( cudaMemcpy(trie_device, trie_reduced, trie_size, cudaMemcpyHostToDevice) ); // Trie for hepfac

    // Set device parameters
    int blocks_per_grid = (current_buffer->length() + threads_per_block -1) / threads_per_block;
    int text_size_stream = current_buffer->length()/count_streams;
    int output_size_stream = result_elements/count_streams;
    int blocks = (blocks_per_grid/count_streams) ? blocks_per_grid/count_streams : 2;
    // split work across multiple streams
    cudaStream_t streams[count_streams];
    for (int i = 0; i < count_streams; ++i){
        //cudaStreamCreate(&streams[i]);
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    std::stringstream kernel_launch;
    kernel_launch << "Launching "<<tests<<"x search_trie_shared<<<" << blocks+1 << ", " << threads_per_block << ",  "<<trie_size<<", streams["<<count_streams<<"]>>>";
    print_info(kernel_launch.str());


    double total_exec_time = 0;
    for (int i=0; i < tests; ++i){
        cudaEventRecord(start);

        {
            int text_offset = 0;
            int output_offset = 0;

            for (int i = 0; i < count_streams-1; ADD1(i)){
                // No need to copy output thanks to new design.
                CUDA_CALL(cudaMemcpyAsync(&text_device[text_offset], &text_host[text_offset], text_size_stream, cudaMemcpyHostToDevice, streams[i]));
                search_trie_shared<<<blocks+1, threads_per_block, trie_size, streams[i]>>>(trie_device, node_reduced_count, text_device, current_buffer->size(), text_offset, output_device);
                CUDA_CALL(cudaMemcpyAsync(&output_host[output_offset], &output_device[output_offset], output_size_stream*sizeof(int), cudaMemcpyDeviceToHost, streams[i]));

                output_offset = (i+1)*output_size_stream;
                text_offset = (i+1)*text_size_stream;
            }

            CUDA_CALL(cudaMemcpyAsync(&text_device[text_offset], &text_host[text_offset], text_size_stream+(current_buffer->length()%count_streams), cudaMemcpyHostToDevice, streams[count_streams-1]));
            search_trie_shared<<<blocks+1, threads_per_block, trie_size, streams[count_streams-1]>>>(trie_device, node_reduced_count, text_device, current_buffer->size(), text_offset, output_device);
            CUDA_CALL(cudaMemcpyAsync(&output_host[output_offset], &output_device[output_offset], (output_size_stream+(result_elements%count_streams))*sizeof(int), cudaMemcpyDeviceToHost, streams[count_streams-1]));

        }
        // streams are non-blocking, without this throughput will measure (incorrectly) 10k+Gbps
        cudaDeviceSynchronize();
        // Stop event timer
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_exec_time += milliseconds;
    }
    std::stringstream timestr;
    timestr << "Average time to execute " << (total_exec_time/tests) << "ms";
    std::stringstream bandwidthstr;
    bandwidthstr << "Average effective bandwidth: " << current_buffer->size()*8/(total_exec_time/tests)/1e6 << "Gb/s";
    print_info(timestr.str());
    print_info(bandwidthstr.str());

    // Fetch results
    int matches = get_matches(output_host);

    // Destroy stream and deallocate memory
    for (int i = 0; i < count_streams; ++i){
        cudaStreamDestroy(streams[i]);
    }

    CUDA_CALL( cudaFree(text_device));
    CUDA_CALL( cudaFree(trie_device));
    CUDA_CALL( cudaFree(output_device));
    CUDA_CALL( cudaFreeHost(text_host));
    CUDA_CALL( cudaFreeHost(output_host));
    // TODO
    return 0;
}

// Texture object Helper
template<typename _type>
cudaTextureObject_t * createTextureObject(_type array1D[], size_t array1D_size){
    cudaTextureObject_t * tex_p = new cudaTextureObject_t();
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = array1D;
    resDesc.res.linear.sizeInBytes = array1D_size;
    resDesc.res.linear.desc = cudaCreateChannelDesc<_type>();
    // Create texture description
    struct cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    CUDA_CALL(cudaCreateTextureObject(tex_p, &resDesc, &texDesc, NULL));
    return tex_p;
}

// Launch Texture-mem-Based Search
int Hepfac::search_texture(){
    print_progress("Launching texture search");

    // Allocate Host
    set_bit_index_size(current_buffer->length());
    unsigned *output_host;
    char *text_host;
    int *trieData;

    CUDA_CALL( cudaMallocHost(&text_host, current_buffer->size()) );
    CUDA_CALL( cudaMemcpy(text_host, current_buffer->c_str(), text_size, cudaMemcpyHostToDevice) ); // Text to be searched
    CUDA_CALL( cudaMallocHost(&output_host, outp_size) );
    memset(output_host, 0, result_elements);
    // Trie is a little more complex
    CUDA_CALL( cudaMallocHost(&trieData,trie_size));
    CUDA_CALL( cudaMemcpy(trieData, trie_reduced, trie_size, cudaMemcpyHostToHost) ); // Trie for hepfac
//    verify_texture_trie(trieData); // no longer really needed


    // Allocate Device
    // Create texture (super easy with templace function)
    cudaTextureObject_t * trie_texture = createTextureObject<int>(trieData, trie_size);
    char *text_device;
    unsigned *output_device;

    CUDA_CALL( cudaMalloc(&text_device, text_size) );
    CUDA_CALL( cudaMalloc(&output_device, outp_size) );

    // Set device parameters
    int blocks_per_grid = (text_size + threads_per_block -1) / threads_per_block;
    int text_size_stream = current_buffer->length()/count_streams;
    int output_size_stream = result_elements/count_streams;
    int blocks = (blocks_per_grid/count_streams) ? blocks_per_grid/count_streams : 2;
    // split work across multiple streams
    cudaStream_t streams[count_streams];
    for (int i = 0; i < count_streams; ++i){
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    std::stringstream kernel_launch;
    kernel_launch << "Launching "<<tests<<"x search_trie_texture<<<" << blocks+1 << ", " << threads_per_block << ",  "<<0<<", streams["<<count_streams<<"]>>>";
    print_info(kernel_launch.str());


    double total_exec_time = 0;
    for (int i=0; i < tests; ++i){
        cudaEventRecord(start);

        {
            int text_offset = 0;
            int output_offset = 0;

            for (int i = 0; i < count_streams-1; ADD1(i)){
                // Copy to device
                CUDA_CALL(cudaMemcpyAsync(&text_device[text_offset], &text_host[text_offset], text_size_stream*sizeof(char), cudaMemcpyHostToDevice, streams[i]));
                // Launch Kernel
                search_trie_texture<<<blocks+1, threads_per_block, 0, streams[i]>>>(*trie_texture, text_device, current_buffer->size(), text_offset, output_device);
                // Copy from device
                CUDA_CALL(cudaMemcpyAsync(&output_host[output_offset], &output_device[output_offset], output_size_stream*sizeof(int), cudaMemcpyDeviceToHost, streams[i]));

                output_offset = (i+1)*output_size_stream;
                text_offset = (i+1)*text_size_stream;
            }

            CUDA_CALL(cudaMemcpyAsync(&text_device[text_offset], &text_host[text_offset], text_size_stream+(current_buffer->length()%count_streams), cudaMemcpyHostToDevice, streams[count_streams-1]));
            search_trie_texture<<<blocks+1, threads_per_block, 0, streams[count_streams-1]>>>(*trie_texture, text_device, current_buffer->size(), text_offset, output_device);
            CUDA_CALL(cudaMemcpyAsync(&output_host[output_offset], &output_device[output_offset], (output_size_stream+(result_elements%count_streams))*sizeof(int), cudaMemcpyDeviceToHost, streams[count_streams-1]));

        }

        // streams are non-blocking, without this throughput will measure (incorrectly) 10k+Gbps
        cudaDeviceSynchronize();
        // Stop event timer
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_exec_time += milliseconds;
		cudaDeviceReset();
    }

    std::stringstream timestr;
    timestr << "Average time to execute " << (total_exec_time/tests) << "ms";
    std::stringstream bandwidthstr;
    bandwidthstr << "Average effective bandwidth: " << current_buffer->size()*8/(total_exec_time/tests)/1e6 << "Gb/s";
    print_info(timestr.str());
    print_info(bandwidthstr.str());

    //  fetch results
    int matches = get_matches(output_host);

    CUDA_CALL( cudaFree(text_device));
    CUDA_CALL( cudaFree(output_device) );
    // CUDA_CALL( cudaFreeHost(trieData) );
    // CUDA_CALL( cudaFreeHost(text_host));
    // CUDA_CALL( cudaFreeHost(output_host));

    // TODO
    return 0;
}



int print_usage(){
    const char * msg = "Usage:\nhepfacpp -p <file:patterns> -t <file:target> -r <int:reduction levels>\nRequired switches: -p, -t\nOptional switches: -r";
    std::cout << msg << std::endl;
    return 1;
}

void fn(int match_index){
    //std::cout << "Match at " << match_index << std::endl;
}

int main(int argc, char ** argv){

    // get patterns and target from file
    std::string filename_patterns = "";
    std::string filename_target = "";
    int reduction_levels = 6,
        loops = 25;

    // CLI handling
    int opt = -1;
    int option_index = 0;
    static struct option long_options[] = {
            {"loops",           optional_argument,  NULL,   'l'},
		    {"help", 			no_argument, 		NULL, 	'h'},
		    {"pattern-file", 	required_argument, 	NULL, 	'p'},
		    {"target-file", 	required_argument, 	NULL, 	't'},
		    {"reductions", 		optional_argument, 	NULL, 	'r'},
		    {NULL,0,NULL,0}
    };

    bool exit_condition = true;

    try {
        while ((opt = getopt_long(argc, argv, "hp:t:r:l:", long_options, &option_index)) != -1){
            // if input is unexpected or doesn't meet requirements
            exit_condition = false;

            if (optarg == "?"){
                return print_usage();
            }

            // parse options
            switch(opt){
                case 'l':
                    loops = (optarg) ? atoi(optarg) : loops;
                    break;
                case 'p':
                    filename_patterns = optarg;
                    break;
                case 't':
                    filename_target = optarg;
                    break;
                case 'r':
                    reduction_levels = (optarg) ? atoi(optarg) : reduction_levels;
                    break;
                default:
                    break;

            }
        }
    } catch (std::exception Ex){
        std::cout << Ex.what() << std::endl;
        return print_usage();
    }
    if (exit_condition){
      return print_usage();
    }

    Hepfac mHepfac(reduction_levels);
    mHepfac.set_test_loops(loops);
    mHepfac.set_match_callback(&fn);
    mHepfac.build_from_file(filename_patterns);    // Build Trie
    mHepfac.set_source_file(filename_target);    // Set target for comparison
    mHepfac.search_global();
    mHepfac.search_shared();                    // Use texture memory for the search
    mHepfac.search_texture();
    //mHepfac.print(mHepfac.identify_results(3));  // print first 3 results (no param for all)
    std::cout << "Just chilling, not exiting because why would you want to exit?" << std::endl;
    return 0;
}
