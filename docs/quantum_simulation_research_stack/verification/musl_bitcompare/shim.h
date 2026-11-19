/* Minimal stand-in for musl's internal libm.h, force-included into every
 * musl source so the files compile standalone with MinGW gcc. Every musl
 * function is renamed with an m_ prefix so it cannot collide with (or be
 * silently replaced by) the host libm. */
#ifndef SHIM_H
#define SHIM_H
#include <stdint.h>
#include <math.h>

#define expm1  m_expm1
#define log1p  m_log1p
#define tanh   m_tanh
#define sinh   m_sinh
#define cosh   m_cosh
#define atanh  m_atanh
#define atan   m_atan
#define atan2  m_atan2
#define asin   m_asin
#define acos   m_acos
#define log10  m_log10
#define hypot  m_hypot
#define exp    m_exp
#define __expo2 m___expo2

double m_expm1(double), m_log1p(double), m_tanh(double), m_sinh(double),
       m_cosh(double), m_atanh(double), m_atan(double), m_atan2(double, double),
       m_asin(double), m_acos(double), m_log10(double), m_hypot(double, double),
       m_exp(double), m___expo2(double, double);

#define FORCE_EVAL(x) do { volatile double __v = (x); (void)__v; } while (0)
#define GET_HIGH_WORD(hi, d) do { union {double f; uint64_t i;} __u; __u.f = (d); (hi) = __u.i >> 32; } while (0)
#define GET_LOW_WORD(lo, d)  do { union {double f; uint64_t i;} __u; __u.f = (d); (lo) = (uint32_t)__u.i; } while (0)
#define EXTRACT_WORDS(hi, lo, d) do { union {double f; uint64_t i;} __u; __u.f = (d); (hi) = __u.i >> 32; (lo) = (uint32_t)__u.i; } while (0)
#define INSERT_WORDS(d, hi, lo) do { union {double f; uint64_t i;} __u; __u.i = ((uint64_t)(hi) << 32) | (uint32_t)(lo); (d) = __u.f; } while (0)
#define SET_LOW_WORD(d, lo) INSERT_WORDS(d, __extension__ ({ uint32_t __h; GET_HIGH_WORD(__h, d); __h; }), lo)
#endif
