#ifndef STAN_MATH_REV_SCAL_HPP
#define STAN_MATH_REV_SCAL_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/meta/is_var.hpp>
#include <stan/math/rev/scal/meta/partials_type.hpp>
#include <stan/math/rev/scal/meta/OperandsAndPartials.hpp>

#include <stan/math/prim/scal.hpp>

#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/acos.hpp>
#include <stan/math/rev/scal/fun/acosh.hpp>
#include <stan/math/rev/scal/fun/as_bool.hpp>
#include <stan/math/rev/scal/fun/asin.hpp>
#include <stan/math/rev/scal/fun/asinh.hpp>
#include <stan/math/rev/scal/fun/atan.hpp>
#include <stan/math/rev/scal/fun/atan2.hpp>
#include <stan/math/rev/scal/fun/atanh.hpp>
#include <stan/math/rev/scal/fun/bessel_first_kind.hpp>
#include <stan/math/rev/scal/fun/bessel_second_kind.hpp>
#include <stan/math/rev/scal/fun/binary_log_loss.hpp>
#include <stan/math/rev/scal/fun/boost_fpclassify.hpp>
#include <stan/math/rev/scal/fun/boost_isfinite.hpp>
#include <stan/math/rev/scal/fun/boost_isinf.hpp>
#include <stan/math/rev/scal/fun/boost_isnan.hpp>
#include <stan/math/rev/scal/fun/boost_isnormal.hpp>
#include <stan/math/rev/scal/fun/calculate_chain.hpp>
#include <stan/math/rev/scal/fun/cbrt.hpp>
#include <stan/math/rev/scal/fun/ceil.hpp>
#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/cosh.hpp>
#include <stan/math/rev/scal/fun/digamma.hpp>
#include <stan/math/rev/scal/fun/erf.hpp>
#include <stan/math/rev/scal/fun/erfc.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/exp2.hpp>
#include <stan/math/rev/scal/fun/expm1.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/falling_factorial.hpp>
#include <stan/math/rev/scal/fun/fdim.hpp>
#include <stan/math/rev/scal/fun/floor.hpp>
#include <stan/math/rev/scal/fun/fma.hpp>
#include <stan/math/rev/scal/fun/fmax.hpp>
#include <stan/math/rev/scal/fun/fmin.hpp>
#include <stan/math/rev/scal/fun/fmod.hpp>
#include <stan/math/rev/scal/fun/frexp.hpp>
#include <stan/math/rev/scal/fun/gamma_p.hpp>
#include <stan/math/rev/scal/fun/gamma_q.hpp>
#include <stan/math/rev/scal/fun/grad_inc_beta.hpp>
#include <stan/math/rev/scal/fun/hypot.hpp>
#include <stan/math/rev/scal/fun/ibeta.hpp>
#include <stan/math/rev/scal/fun/if_else.hpp>
#include <stan/math/rev/scal/fun/inc_beta.hpp>
#include <stan/math/rev/scal/fun/inv.hpp>
#include <stan/math/rev/scal/fun/inv_cloglog.hpp>
#include <stan/math/rev/scal/fun/inv_logit.hpp>
#include <stan/math/rev/scal/fun/inv_Phi.hpp>
#include <stan/math/rev/scal/fun/inv_sqrt.hpp>
#include <stan/math/rev/scal/fun/inv_square.hpp>
#include <stan/math/rev/scal/fun/is_inf.hpp>
#include <stan/math/rev/scal/fun/is_nan.hpp>
#include <stan/math/rev/scal/fun/is_uninitialized.hpp>
#include <stan/math/rev/scal/fun/ldexp.hpp>
#include <stan/math/rev/scal/fun/lgamma.hpp>
#include <stan/math/rev/scal/fun/lmgamma.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log10.hpp>
#include <stan/math/rev/scal/fun/log1m.hpp>
#include <stan/math/rev/scal/fun/log1m_exp.hpp>
#include <stan/math/rev/scal/fun/log1p.hpp>
#include <stan/math/rev/scal/fun/log1p_exp.hpp>
#include <stan/math/rev/scal/fun/log2.hpp>
#include <stan/math/rev/scal/fun/log_diff_exp.hpp>
#include <stan/math/rev/scal/fun/log_falling_factorial.hpp>
#include <stan/math/rev/scal/fun/log_mix.hpp>
#include <stan/math/rev/scal/fun/log_rising_factorial.hpp>
#include <stan/math/rev/scal/fun/log_sum_exp.hpp>
#include <stan/math/rev/scal/fun/modified_bessel_first_kind.hpp>
#include <stan/math/rev/scal/fun/modified_bessel_second_kind.hpp>
#include <stan/math/rev/scal/fun/multiply_log.hpp>
#include <stan/math/rev/scal/fun/owens_t.hpp>
#include <stan/math/rev/scal/fun/Phi.hpp>
#include <stan/math/rev/scal/fun/Phi_approx.hpp>
#include <stan/math/rev/scal/fun/pow.hpp>
#include <stan/math/rev/scal/fun/primitive_value.hpp>
#include <stan/math/rev/scal/fun/rising_factorial.hpp>
#include <stan/math/rev/scal/fun/round.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>
#include <stan/math/rev/scal/fun/sinh.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/square.hpp>
#include <stan/math/rev/scal/fun/squared_distance.hpp>
#include <stan/math/rev/scal/fun/step.hpp>
#include <stan/math/rev/scal/fun/tan.hpp>
#include <stan/math/rev/scal/fun/tanh.hpp>
#include <stan/math/rev/scal/fun/to_var.hpp>
#include <stan/math/rev/scal/fun/tgamma.hpp>
#include <stan/math/rev/scal/fun/trunc.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>

#endif
