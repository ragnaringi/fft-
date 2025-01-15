/*
MIT License

Copyright (c) 2024 Ragnar Hrafnkelsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <cassert>
#include <cstring>
#include "FFTComplex.h"

namespace fftpp
{

template <typename T, typename Alloc = std::allocator<T>>
class FFTReal
{
public:
    //==========================================================================
    FFTReal (size_t size, const Alloc& alloc = Alloc());
    
    void forward (const T* timeData, std::complex<T>* freqData);
    void inverse (const std::complex<T>* freqData, T* timeData, bool inPlace = true);
    
    size_t getSize() const noexcept      { return size * 2; }
    
protected:
    //==========================================================================
    const size_t size;
    FFTComplex<T, Alloc> fft;
    std::vector<T, Alloc> twiddles;
    std::complex<T>* twiddlesCplx;
};


//==============================================================================
//
//==============================================================================

template <typename T, typename Alloc>
FFTReal<T, Alloc>::FFTReal (size_t fftSize, const Alloc& alloc)
  : size (halve (fftSize)), fft (size, alloc), twiddles (alloc)
{
    assert ((size & 1) == 0 && "Real FFT size must be even.");
    
    twiddles.resize (size * 2);
    
    twiddlesCplx = reinterpret_cast<std::complex<T>*> (twiddles.data());
    
    for (auto i = 0; i < size; ++i)
    {
        const double phase = -3.141592653589793238L * ((double) (i + 1) / size + 0.5);
        cexp (twiddlesCplx + i, phase);
    }
}

template <typename T, typename Alloc>
void FFTReal<T, Alloc>::forward (const T* timeData, std::complex<T>* freqData)
{
    fft.forward (timeData, freqData);
    
    if constexpr (fftpp_is_integral<T>)
    {
        for (auto k = 0; k < size; ++k)
            cdiv (freqData[k], 2);
    }
    
    auto tdc = freqData[0];
    freqData[0]    = { tdc.real() + tdc.imag(), 0 };
    freqData[size] = { tdc.real() - tdc.imag(), 0 };
    
    for (auto k = 1; k <= size / 2; ++k)
    {
        auto s0 = freqData[k];
        auto s1 = std::conj (freqData[size - k]);
        auto fk   = s0 + s1;
        auto fknc = s0 - s1;
        auto tw = cmul (fknc, twiddlesCplx[k - 1]);
        
        freqData[k]        = { halve (fk.real() + tw.real()),
                               halve (fk.imag() + tw.imag()) };
        freqData[size - k] = { halve (fk.real() - tw.real()),
                               halve (tw.imag() - fk.imag()) };
    }
    
    // Clear negative frequencies
    std::memset (freqData + size, 0, size * sizeof (std::complex<T>));
}

template <typename T, typename Alloc>
void FFTReal<T, Alloc>::inverse (const std::complex<T>* freqData, T* timeData, bool inPlace)
{
    auto* tempBuffer = const_cast<std::complex<T>*> (freqData);
    
    if (! inPlace)
    {
        tempBuffer = (std::complex<T>*) alloca (size * sizeof (std::complex<T>));
        
        tempBuffer[0] = { freqData[0].real() + freqData[size].real(),
                          freqData[0].real() - freqData[size].real() };
        std::memcpy (tempBuffer + 1, freqData + 1, (size - 1) * sizeof (std::complex<T>));
    }
    
    if constexpr (fftpp_is_integral<T>)
    {
        for (auto k = 0; k < size; k++)
            cdiv (tempBuffer[k], 2);
    }
    
    for (auto k = 1; k <= size / 2; k++)
    {
        auto s0 = tempBuffer[k];
        auto s1 = std::conj (tempBuffer[size - k]);
        auto fk   = s0 + s1;
        auto fknc = s0 - s1;
        auto tw = cmul (fknc, std::conj (twiddlesCplx[k - 1]));
        
        tempBuffer[k]        = fk + tw;
        tempBuffer[size - k] = std::conj (fk - tw);
    }
    
    fft.inverse (tempBuffer, timeData);
}

} // namespace fftpp
