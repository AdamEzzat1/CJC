#include <math.h>
/* Not a musl transcription target: musl's exp is table-driven. Paths using it are reported separately. */
double m_exp(double x) { return exp(x); }
