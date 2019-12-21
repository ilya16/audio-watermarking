# Audio Watermarking using DWT/DFT and SVD

Implementation of the SVD-based audio watermarking schemes.

Final project in Numerical Linear Algebra [course](https://nla.skoltech.ru) at Skoltech. Done by I. Borovik, K. Pongsirijinda, and J. Atorigo.

See [demo](demo.ipynb) for the demonstration and comparison of the following approaches:

* Time Domain + SVD
* 1-D Discrete Wavelet Transform (DWT) + SVD
* 2-D Discrete Wavelet Transform (DWT2) + SVD
* 1-D Discrete Fourier Transform (DFT) + SVD
* 2-D Discrete Fourier Transform (DFT2) + SVD

Watermarking schemes are compared based on:
* the quality of the watermarked signals
* robustness to audio modification attacks (addition of Gaussian noise, audio cropping and scaling)


