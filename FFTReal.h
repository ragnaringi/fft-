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

template <typename T>
class FFTReal
{
public:
    //==========================================================================
    FFTReal (size_t size);
    
    void forward (const T* timeData, std::complex<T>* freqData);
    void inverse (const std::complex<T>* freqData, T* timeData);
    
    size_t getSize() const noexcept      { return size * 2; }

protected:
    //==========================================================================
    const size_t size;
    FFTComplex<T> fft;
    std::vector<std::complex<T>> twiddlesFwd, twiddlesInv;
};


//==============================================================================
//
//==============================================================================
template <typename T>
static void initTwiddleTable (std::vector<std::complex<T>>& twiddles, const size_t size, const int inverse)
{
    twiddles.resize (size);

    for (int i = 0; i < size; ++i)
    {
        const double phase = -3.14159265358979323846264338327 * ((double) (i + 1) / size + 0.5);
        cexp (twiddles.data() + i, phase * inverse);
    }
}

template <typename T>
FFTReal<T>::FFTReal (size_t fftSize)
  : size (halve (fftSize)), fft (size)
{
    assert ((size & 1) == 0 && "Real FFT size must be even.");

    initTwiddleTable (twiddlesFwd, size,  1);
    initTwiddleTable (twiddlesInv, size, -1);
}

template <typename T>
void FFTReal<T>::forward (const T* timeData, std::complex<T>* freqData)
{
    auto* tmpbuf = (std::complex<T>*) alloca (size * sizeof (std::complex<T>));

    fft.forward (timeData, tmpbuf);

    if constexpr (std::is_integral_v<T>)
    {
        for (auto k = 0; k < size; ++k)
            cdiv (tmpbuf[k], 2);
    }

    auto tdc = tmpbuf[0];
    freqData[0]    = { tdc.real() + tdc.imag(), 0 };
    freqData[size] = { tdc.real() - tdc.imag(), 0 };

    for (auto k = 1; k <= size / 2; ++k)
    {
        auto s0 = tmpbuf[k];
        auto s1 = std::conj (tmpbuf[size - k]);
        auto fk   = s0 + s1;
        auto fknc = s0 - s1;
        auto tw = cmul (fknc, twiddlesFwd[k - 1]);

        freqData[k]        = { halve (fk.real() + tw.real()),
                               halve (fk.imag() + tw.imag()) };
        freqData[size - k] = { halve (fk.real() - tw.real()),
                               halve (tw.imag() - fk.imag()) };
    }
}

template <typename T>
void FFTReal<T>::inverse (const std::complex<T>* freqData, T* timeData)
{
    auto* tmpbuf = (std::complex<T>*) alloca (size * sizeof (std::complex<T>));

    tmpbuf[0] = { freqData[0].real() + freqData[size].real(),
                  freqData[0].real() - freqData[size].real() };
    std::memcpy (tmpbuf + 1, freqData + 1, (size - 1) * sizeof (std::complex<T>));

    if constexpr (std::is_integral_v<T>)
    {
        for (auto k = 0; k < size; k++)
            cdiv (tmpbuf[k], 2);
    }

    for (auto k = 1; k <= size / 2; k++)
    {
        auto s0 = tmpbuf[k];
        auto s1 = std::conj (tmpbuf[size - k]);
        auto fk   = s0 + s1;
        auto fknc = s0 - s1;
        auto tw = cmul (fknc, twiddlesInv[k - 1]);

        tmpbuf[k]        = fk + tw;
        tmpbuf[size - k] = std::conj (fk - tw);
    }

    fft.inverse (tmpbuf, timeData);
}
