# HEPFAC_CPP

A C++ implementation of the HEPFAC Algorithm.

## Getting Started

This implementation includes almost everything needed to get started with HEPFAC. It is worth noting however, that some functions are application dependant and should be overridden.  

Otherwise, I've attempted to make this as easy to use as possible; due to the object oriented approach multiple Trie arrays can exist at once (as separate *Hepfac* objects), which should aid testing. Additionally, there are default methods for signature/pattern file reading, and target data file reading, among others.

### Prerequisites

As part of my goal to make this as easy to use as possible, the only external dependency is cuda. On Linux (tested with Fedora), it is named **cuda-devel**. There is one internal dependency not currently included on the repo - DetailedException, this can be 'faked' with the following code:  

```
class DetailedException : std::runtime_error{
    DetailedException() throw() : std::runtime_error() {}
}

namespace MYNAMESPACE{
    class SomeErrorMsg : DetailedException {}
}
```  

DetailedException will likely be removed in the next version!

## Usage

```
    int reduction_size = 6;                                 // depth of characters to search  
    Hepfac myHepfac(reduction_size);                        // MUST Construct with reduction_size (for now)

    myHepfac.query_device();                                // Query device specifications [Optional]            

    myHepfac.build_from_file(filename_patterns);            // if using build from file, pass your filename here
    myHepfac.set_source_type(Hepfac::SourceType::FILE);     // set target data source type (FILE is only implementation provided) [Optional]
    myHepfac.set_source_file(filename_target);              // set target data source filename
    myHepfac.search_global();                               // perform hepfac search with global memory (or shared, texture, hybrid)

```

## Built With

* [CUDA](https://docs.nvidia.com/cuda/) - GPGPU programming interface model created by Nvidia


## Contributing

Please read [CONTRIBUTING.md](#None) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors
**Andrew Calder** - *HEPFAC_CPP* - [AR-Calder](https://github.com/AR-Calder)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

**Dr Xavier Bellekens** - *HEPFAC_C* - [Noktec](https://github.com/Noktec)
