/* Private helpers shared by the C translation units. Not installed. */
#ifndef QSL_INTERNAL_H
#define QSL_INTERNAL_H

#include "qsl/qsl.h"

#define QSL_INV_SQRT_2PI 0.39894228040143267794 /* 1/sqrt(2*pi) */
#define QSL_SQRT_2PI 2.50662827463100050242     /* sqrt(2*pi)   */
#define QSL_SQRT1_2 0.70710678118654752440      /* 1/sqrt(2)    */

static inline int qsl_valid_type(qsl_option_type type) {
    return type == QSL_CALL || type == QSL_PUT;
}

#endif /* QSL_INTERNAL_H */
