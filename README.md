# smalltopk

Smalltopk library is designed to speed up the training for `Product Quantization` and `Product Residual Quantization`. An [long article](article/main5.md) about the library is available.

This library was written specifically for the Milvus project (https://github.com/milvus-io/milvus) and is planned to be part of it.

This is an early alpha version for both the code and [the article](article/main5.md).

Tested on:
* Intel Xeon 4-th gen (Sapphire Rapids 8488C)
* AMD Zen 4 (9R14)
* AWS Graviton 3

# Building the library

Please refer to the following [section](article/main5.md#building-the-library).

Overall, the only used external library is `OpenMP` for a very basic multithreading of a single block. Thus, `OpenMP` can be easily replaced with any other threading facility, including the one from the standard C++ library (which I did not use, bcz I was not sure about a possible thread pool under the hood).

# Integration with FAISS

Please refer to the following [section](article/main5.md#integration-with-faiss).

FAISS library can be found on [github](https://github.com/facebookresearch/faiss).

Benchmarks for [Product Quantizer](article/main5.md#benchmarks-for-product-quantizater) and [Product Residual Quantizer](article/main5.md#benchmarks-for-product-residual-quantizer).

# Unit tests

Unit tests use reworked yet borrowed code from FAISS.
