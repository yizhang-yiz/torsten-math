#ifndef STAN_MATH_PRIM_ARR_FUNCTOR_INTEGRATE_UNIVARIATE_HPP
#define STAN_MATH_PRIM_ARR_FUNCTOR_INTEGRATE_UNIVARIATE_HPP

//#include <Eigen/Dense>
#include <stan/math/prim/arr/functor/integrate_ode_rk45.hpp>
//#include <stan/math/rev/mat/functor/integrate_ode_bdf.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace stan {
  namespace math {

    /**
     * Return the value of the integral of a univariate function 
     * over a bounded interval given integrand as a Stan function 
     * and the bounds of the integration interval.
     *
     * @tparam F type of integrand function.
     * @tparam T0 Type of scalar for t0.
     * @tparam T1 Type of scalar for t1.
     * @tparam T2 type of scalars for parameters.
     * @param[in] f functor for the integrand function.
     * @param[in] x0 lower integration bound.
     * @param[in] x1 upper integration bound.
     * @param[in] theta parameter vector for the integrand.
     * @param[in] x continuous data vector for the integrand.
     * @param[in] x_int integer data vector for the integrand.
     * @param[out] msgs the print stream for warning messages.
     * @param[in] relative_tolerance relative tolerance parameter
     *   for Boost's ode solver. Defaults to 1e-6.
     * @param[in] absolute_tolerance absolute tolerance parameter
     *   for Boost's ode solver. Defaults to 1e-6.
     * @param[in] max_num_steps maximum number of steps to take within
     *   the Boost ode solver.
     * @return a scalar value of the integration result.
     */

    template <typename T0, typename T1, typename T2>
    typename boost::math::tools::promote_args <T0, T1, T2>::type
    linear_interpolation(const T0& xout,
                         const std::vector<T1>& x,
                         const std::vector<T2>& y) {
      typedef typename boost::math::tools::promote_args <T0, T1, T2>::type
        scalar;
      using std::vector;
      int nx = x.size();
      scalar yout;

      check_finite("linear_interpolation", "xout", xout);
      check_finite("linear_interpolation", "x", x);
      check_finite("linear_interpolation", "y", y);
      check_nonzero_size("linear_interpolation", "x", x);
      check_nonzero_size("linear_interpolation", "y", y);
      check_ordered("linear_interpolation", "x", x);
      check_matching_sizes("linear_interpolation", "x", x, "y", y);

      if (xout <= x[0]) {
        yout = y[0];
      } else if (xout >= x[nx - 1]) {
        yout = y[nx - 1];
      } else {
        int i = SearchReal(x, nx, xout) - 1;
        yout = y[i] + (y[i+1] - y[i]) / (x[i+1] - x[i]) * (xout - x[i]);
      }

      return yout;
    }

    template <typename T0, typename T1, typename T2>
    std::vector <typename boost::math::tools::promote_args <T0, T1, T2>::type>
    linear_interpolation(const std::vector<T0>& xout,
                         const std::vector<T1>& x,
                         const std::vector<T2>& y) {
      typedef typename boost::math::tools::promote_args <T0, T1, T2>::type
        scalar;
      using std::vector;

      int nxout = xout.size();
      vector<scalar> yout(nxout);

      check_nonzero_size("linear_interpolation", "xout", xout);
      check_finite("linear_interpolation", "xout", xout);

      for (int i = 0; i < nxout; i++) {
        yout[i] = linear_interpolation(xout[i], x, y);
      }
      return yout;
    }

  }
}
#endif
