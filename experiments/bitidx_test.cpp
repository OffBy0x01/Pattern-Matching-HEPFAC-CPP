#include "../lib/detailed_exception/DetailedException.cpp"
#include <functional>
#include <array>
#include <iostream>

template <typename _type, unsigned int _bitsize>
struct bit_index {
    typedef _type               value_type;
    typedef value_type&         reference;
    typedef unsigned int        size_type;

    const unsigned int          block_size = sizeof(value_type)*8;
    //http://supertech.csail.mit.edu/papers/debruijn.pdf
    const int debruijn32precomputed[32] = {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
    };

    // TODO use C array
    static constexpr size_type
    array_size = (_bitsize / (sizeof(value_type)*8)) + !!(_bitsize & ((sizeof(value_type)*8)-1));

    std::array<value_type, array_size> data;

    std::function<void(int)> match_found_callback;

    // Capacity
    constexpr size_type
    size() const noexcept { return data.size(); }

    constexpr size_type
    bit_size() const noexcept { return _bitsize; }

    constexpr bool
    empty() const noexcept { return 0 == size(); }

    // Callback
    void
    set_callback(std::function<void(int)> callback_func){
        match_found_callback = callback_func;
    }

    // Helpers
    constexpr size_type
    util_popc(size_type temp){
        // This is the fastest popcount for 32-bit types as detailed in "Software Optimization Guide for AMD64 Processors"(179-180)
        temp = temp - ((temp >> 1) & 0b01010101010101010101010101010101);
        temp = (temp & 0b00110011001100110011001100110011) + ((temp >> 2) & 0b00110011001100110011001100110011);
        return (((temp + (temp >> 4)) & 0b00001111000011110000111100001111) * 0b00000001000000010000000100000001) >> 24;
    }

    constexpr size_type
    util_popcto(size_type& temp, size_type& idx){
        const int bmi = idx >> 5;       // max data index
              int count = 0;            // number of set bits

        for (int i = 0; i < bmi; ++i){
            count += util_popc(data[i]);
        }
        // This uses the same popcount as above but masks to the required index
        return count + util_popc(data[bmi] & ((1<<idx)-1));
    }

    constexpr size_type
    popcto(size_type& idx){
        return util_popcto(data, idx);
    }

    constexpr size_type
    get_msb_val(size_type& v){
        // set all zero bits right of msb to 1.
        size_type vmin = v-1;
        vmin |= vmin >> 1;
        vmin |= vmin >> 2;
        vmin |= vmin >> 4;
        vmin |= vmin >> 8;
        vmin |= vmin >> 16;
        // raise to next msb then reduce if not original msb.
        return (vmin+1)>> !!(v & vmin);
    }

    constexpr size_type
    get_msb_idx(size_type& v){
        // This is disgusting and I hate it, but it works.
        const int msb_val = get_msb_val(v);
        return debruijn32precomputed[((msb_val & -msb_val)*0x077CB531UL) >> 27];
    }

    constexpr size_type
    get_lsb_idx(size_type& v){
        return debruijn32precomputed[((v & -v)*0x077CB531UL) >> 27];
    }

    constexpr size_type
    get_lsb_val(size_type& v){
        return v & -v;
    }

    // Get
    constexpr size_type
    get_idx(size_type idx) {
        if (idx >= _bitsize)
            throw GENERIC::OutOfRange();
        return data[idx/(8*sizeof(value_type))] & 1 << (idx & block_size -1);
    }

    constexpr size_type
    get_all() {
        // must have set a lambda or function reference as callback_func
        for (int i=0; i<data.size(); ++i){
            if(data[i]){
                for(int j=get_lsb_idx(data[i]); j<block_size; ++j){
                    if (data[i] & 1 << (j &  block_size -1)){
                        match_found_callback((i*block_size)+j);
                    }
                }
            }
        }
    }

    // Set
    constexpr void
    set_idx(size_type idx) {
        if (idx >= _bitsize)
            throw GENERIC::OutOfRange();
        data[idx/block_size] |= 1 << (idx & (block_size -1));
    }

    constexpr void
    set_all(size_type idx) {
        if (idx >= _bitsize)
            throw GENERIC::OutOfRange();
        data[idx/block_size] |= ((1 << (block_size - 1)) | ~(1 << (block_size - 1)));
    }

    // Unset
    constexpr void
    unset_idx(size_type idx) {
        if (idx >= _bitsize)
            throw GENERIC::OutOfRange();
        if (data[idx/block_size] & 1 << (idx & block_size-1))
            data[idx/block_size] ^= 1 << (idx & block_size-1);
    }

    constexpr void
    unset_all(size_type idx) {
        if (idx >= _bitsize)
            throw GENERIC::OutOfRange();
        data[idx/block_size] &= 0;
    }
};


void fn(int location){
    std::cout << "Found at: " << location << std::endl;
}

int main(){
    const int max = 33;
    bit_index<unsigned, max> example;
    std::cout << "bit_size="<< example.bit_size() <<", byte_size=" << example.size() << std::endl;
    example.set_callback(&fn);
    std::cout << "Setting callback "<< std::endl;
    example.set_callback(&fn);
    std::cout << "Setting Bits "<< std::endl;
    example.set_all(max -1);
    std::cout << "Getting Bits "<< std::endl;
    example.get_all();



}
