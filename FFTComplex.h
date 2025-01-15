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

#include <complex>
#include <vector>
#include <type_traits>

namespace fftpp
{

template <typename T, typename Alloc = std::allocator<T>>
class FFTComplex
{
public:
    //==========================================================================
    FFTComplex (size_t size, const Alloc& alloc = Alloc());
    
    void forward (const T* timeData, std::complex<T>* freqData);
    void inverse (const std::complex<T>* freqData, T* timeData);
    
    size_t getSize() const noexcept      { return size; }
    
protected:
    //==========================================================================
    struct Factor { size_t radix, length; };
    
    void perform (const std::complex<T>*, std::complex<T>*, const size_t, int, Factor*, bool);
    void butterfly2 (std::complex<T>*, const size_t, const size_t, std::complex<T>*);
    void butterfly4 (std::complex<T>*, const size_t, const size_t, std::complex<T>*, bool);
    void butterflyGeneric (std::complex<T>*, const size_t, const size_t, const size_t, std::complex<T>*);
    
    const size_t size;
    Factor factors[32];
    std::vector<T, Alloc> twiddlesFwd, twiddlesInv;
};


//==============================================================================
//
//==============================================================================

#ifndef __cpp_lib_type_trait_variable_templates
template <typename T>
constexpr bool fftpp_is_floating_point = std::is_floating_point<T>::value;
template <typename T>
constexpr bool fftpp_is_integral       = std::is_integral<T>::value;
#else
template <typename T>
constexpr bool fftpp_is_floating_point = std::is_floating_point_v<T>;
template <typename T>
constexpr bool fftpp_is_integral       = std::is_integral_v<T>;
#endif

// Scalar math functions
template <typename T>
T sround (T x)
{
    static_assert (! fftpp_is_floating_point<T>, "type can't be float");
    static constexpr T FRACBITS = 31;
    return (T) (x + (1 << (FRACBITS - 1))) >> FRACBITS;
}

template <typename T>
static inline T scos (double phase)
{
    if constexpr (fftpp_is_floating_point<T>)
        return std::cos (phase);
    else
        return std::floor (0.5 + std::numeric_limits<T>::max() * std::cos(phase));
}

template <typename T>
static inline T ssin (double phase)
{
    if constexpr (fftpp_is_floating_point<T>)
        return std::sin (phase);
    else
        return floor (0.5 + std::numeric_limits<T>::max() * std::sin (phase));
}

template <typename T>
static inline T smul (T a, T b)
{
    if constexpr (fftpp_is_floating_point<T>)
        return a * b;
    else
        return (T) sround ((int64_t) a * (int64_t) b);
}

template <typename T>
static inline T sdiv (T a, T b)
{
    if constexpr (fftpp_is_floating_point<T>)
        return a / b;
    else
        return smul (a, std::numeric_limits<T>::max() / b);
}

template <typename T>
T halve (T x)
{
    if constexpr (fftpp_is_floating_point<T>)
        return x * T (0.5);
    else
        return x >> 1;
}

// Complex math functions
template <typename T>
static inline std::complex<T> cmul (const std::complex<T>& a, const std::complex<T>& b)
{
    return { smul (a.real(), b.real()) - smul (a.imag(), b.imag()),
             smul (a.real(), b.imag()) + smul (a.imag(), b.real()) };
}

template <typename T, typename D>
static inline void cdiv (std::complex<T>& c, D d)
{
    c.real (sdiv (c.real(), (T) d));
    c.imag (sdiv (c.imag(), (T) d));
}

template <typename T, typename P = double>
static inline void cexp (std::complex<T>* x, P phase)
{
    x->real (scos<T> (phase));
    x->imag (ssin<T> (phase));
}

//==============================================================================
template <typename T, typename Alloc>
FFTComplex<T, Alloc>::FFTComplex (size_t fftSize, const Alloc& alloc)
  : size (fftSize), twiddlesFwd (alloc), twiddlesInv (alloc)
{
    twiddlesFwd.resize (size * 2);
    twiddlesInv.resize (size * 2);
      
    const double pi = 3.141592653589793238L;
    const double factor = -2 * pi / size;
    
    auto* twiddlesFwdCplx = reinterpret_cast<std::complex<T>*> (twiddlesFwd.data());
    auto* twiddlesInvCplx = reinterpret_cast<std::complex<T>*> (twiddlesInv.data());
    
    for (auto i = 0; i < size; ++i)
    {
        cexp (twiddlesFwdCplx + i, factor * i);
        cexp (twiddlesInvCplx + i, factor * i * -1);
    }
    
    size_t p = 4;
    size_t root = std::sqrt ((double) size);
    Factor* factorsPtr = factors;
    
    do
    {
        while (fftSize % p)
        {
            switch (p)
            {
                case 4:  p = 2; break;
                case 2:  p = 3; break;
                default: p += 2; break;
            }
            
            if (p > root)
                p = fftSize;
        }
        
        fftSize /= p;
        
        auto& factor = *factorsPtr++;
        factor.radix = p;
        factor.length = fftSize;
    }
    while (fftSize > 1);
}

template <typename T, typename Alloc>
void FFTComplex<T, Alloc>::forward (const T* timeData, std::complex<T>* freqData)
{
    perform (reinterpret_cast<const std::complex<T>*> (timeData), freqData, 1, 1, factors, false);
}

template <typename T, typename Alloc>
void FFTComplex<T, Alloc>::inverse (const std::complex<T>* freqData, T* timeData)
{
    perform (freqData, reinterpret_cast<std::complex<T>*> (timeData), 1, 1, factors, true);
}

template <typename T, typename Alloc>
void FFTComplex<T, Alloc>::perform (const std::complex<T>* input, std::complex<T>* output,
                                    const size_t stride, int inStride,
                                    Factor* factors, bool inverse)
{
    const auto& factor = *factors++;
    const auto radix  = factor.radix;
    const auto length = factor.length;
    
    auto* outBegin = output;
    const auto* outEnd = outBegin + radix * length;
    
    if (length == 1)
    {
        do
        {
            *output = *input;
            input  += stride * inStride;
        }
        while (++output != outEnd);
    }
    else
    {
        do
        {
            perform (input, output, stride * radix, inStride, factors, inverse);
            input += stride * inStride;
        }
        while ((output += length) != outEnd);
    }
    
    output = outBegin;
    
    auto* twiddles = reinterpret_cast<std::complex<T>*> (inverse ? twiddlesInv.data() : twiddlesFwd.data());
    
    switch (radix)
    {
        case 2:  butterfly2 (output, stride, length, twiddles); break;
        case 4:  butterfly4 (output, stride, length, twiddles, inverse); break;
        default: butterflyGeneric (output, stride, radix, length, twiddles); break;
    }
}

template <typename T, typename Alloc>
void FFTComplex<T, Alloc>::butterfly2 (std::complex<T>* output, const size_t stride,
                                       const size_t length, std::complex<T>* twiddles)
{
    auto* output2 = output + length;
    
    for (auto i = 0; i < length; ++i)
    {
        if constexpr (fftpp_is_integral<T>)
        {
            cdiv (*output,  2);
            cdiv (*output2, 2);
        }
        
        auto t = cmul (*output2, *twiddles);
        twiddles += stride;
        
        (*output2++) = (*output) - t;
        (*output++) += t;
    }
}

template <typename T, typename Alloc>
void FFTComplex<T, Alloc>::butterfly4 (std::complex<T>* output, const size_t stride,
                                       const size_t length, std::complex<T>* twiddles,
                                       bool inverse)
{
    const auto* outEnd = output + length;
    
    const size_t length2 = 2 * length;
    const size_t length3 = 3 * length;
    
    if constexpr (fftpp_is_integral<T>)
    {
        do
        {
            cdiv (output[length],  4);
            cdiv (output[length2], 4);
            cdiv (output[length3], 4);
            cdiv (*output++, 4);
        }
        while (output != outEnd);
        
        output = output - length;
    }
    
    std::complex<T> *tw1, *tw2, *tw3;
    tw3 = tw2 = tw1 = twiddles;
    
    do
    {
        auto s0 = cmul (output[length],  *tw1);
        auto s1 = cmul (output[length2], *tw2);
        auto s2 = cmul (output[length3], *tw3);
        auto s3 = s0 + s2;
        auto s4 = s0 - s2;
        auto s5 = (*output) - s1;
        
        (*output) += s1;
        output[length2] = (*output) - s3;
        (*output) += s3;
        
        if (inverse)
        {
            output[length]  = { s5.real() - s4.imag(),
                                s5.imag() + s4.real() };
            output[length3] = { s5.real() + s4.imag(),
                                s5.imag() - s4.real() };
        }
        else
        {
            output[length]  = { s5.real() + s4.imag(),
                                s5.imag() - s4.real() };
            output[length3] = { s5.real() - s4.imag(),
                                s5.imag() + s4.real() };
        }
        
        tw1 += stride;
        tw2 += stride * 2;
        tw3 += stride * 3;
    }
    while (++output != outEnd);
}

template <typename T, typename Alloc>
void FFTComplex<T, Alloc>::butterflyGeneric (std::complex<T>* output, const size_t stride,
                                             const size_t radix, const size_t length,
                                             std::complex<T>* twiddles)
{
    auto* scratch = (std::complex<T>*) alloca (sizeof (std::complex<T>) * radix);
    
    if constexpr (fftpp_is_integral<T>)
    {
        for (auto u = 0; u < length; ++u)
        {
            for (int k = u, q1 = 0; q1 < radix; ++q1)
            {
                cdiv (output[k], radix);
                k += length;
            }
        }
    }
    
    for (auto u = 0; u < length; ++u)
    {
        for (auto k = u, q1 = 0; q1 < radix; ++q1)
        {
            scratch[q1] = output[k];
            k += length;
        }
        
        for (auto k = u, q1 = 0; q1 < radix; ++q1)
        {
            output[k] = scratch[0];
            
            for (auto twIndex = 0, q = 1; q < radix; ++q)
            {
                twIndex += stride * k;
                
                if (twIndex >= size)
                    twIndex -= size;
                
                output[k] += cmul (scratch[q], twiddles[twIndex]);
            }
            
            k += length;
        }
    }
}

} // namespace fftpp
