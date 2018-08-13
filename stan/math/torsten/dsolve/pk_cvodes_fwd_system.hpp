#ifndef STAN_MATH_TORSTEN_DSOLVE_CVODES_FWD_SYSTEM_HPP
#define STAN_MATH_TORSTEN_DSOLVE_CVODES_FWD_SYSTEM_HPP

#include <stan/math/prim/arr/fun/value_of.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/torsten/dsolve/pk_cvodes_system.hpp>
#include <stan/math/torsten/dsolve/cvodes_sens_rhs.hpp>
#include <stan/math/torsten/pk_csda.hpp>
#include <stan/math/rev/mat/functor/jacobian.hpp>
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <ostream>
#include <vector>

namespace torsten {
  namespace dsolve {

    /**
     * CVODES ODE system with forward sensitivity calculation
     *
     * @tparam F type of functor for ODE residual.
     * @tparam Ty0 type of initial unknown values.
     * @tparam Tpar type of parameters.
     */
    template <typename F, typename Tts, typename Ty0, typename Tpar, int Lmm>
    class pk_cvodes_fwd_system :
      public pk_cvodes_system<F, Tts, Ty0, Tpar, Lmm> {
    public:
      using Ode = pk_cvodes_system<F, Tts, Ty0, Tpar, Lmm>;
    private:
      N_Vector* nv_ys_;
      std::vector<std::complex<double> >& yy_cplx_;
      std::vector<std::complex<double> >& theta_cplx_;
      std::vector<std::complex<double> >& fval_cplx_;
    public:
      /**
       * Construct CVODES ODE system from initial condition and parameters
       *
       * @param[in] f ODE residual functor
       * @param[in] y0 initial condition
       * @param[in] theta parameters of the base ODE
       * @param[in] x_r continuous data vector for the ODE
       * @param[in] x_i integer data vector for the ODE
       * @param[in] msgs stream to which messages are printed
       */
      pk_cvodes_fwd_system(cvodes_service<Ode>& serv,
                           const F& f,
                           double t0,
                           const std::vector<Tts>& ts,
                           const std::vector<Ty0>& y0,
                           const std::vector<Tpar>& theta,
                           const std::vector<double>& x_r,
                           const std::vector<int>& x_i,
                           std::ostream* msgs) :
        Ode(serv, f, t0, ts, y0, theta, x_r, x_i, msgs),
        nv_ys_(Ode::serv_.nv_ys),
        yy_cplx_(Ode::serv_.yy_cplx),
        theta_cplx_(Ode::serv_.theta_cplx),
        fval_cplx_(Ode::serv_.fval_cplx)
      {}

      /**
       * Dummy destructor. Deallocation of CVODES memory is done
       * in @c cvodes_service.
       */
      ~pk_cvodes_fwd_system() {
      }

      /**
       * return N_Vector pointer array of sensitivity
       */
      N_Vector* nv_ys() { return nv_ys_; }

      /**
       * convert to void pointer for CVODES callbacks
       */
      void* to_user_data() {  // prepare to inject ODE info
        return static_cast<void*>(this);
      }

      /**
       * Calculate sensitivity rhs using CVODES vectors. The
       * internal workspace is allocated by @c cvodes_service.
       * We use CSDA to compute senstivity, so we need to
       * generate complex version of parameters.
       */
      void eval_sens_rhs(int ns, double t, N_Vector y, N_Vector ydot,
                         N_Vector* ys, N_Vector* ysdot,
                         N_Vector temp1, N_Vector temp2) {
        using std::complex;
        using cplx = complex<double>;
        using B = pk_cvodes_system<F, Tts, Ty0, Tpar, Lmm>;
        const int n = B::N_;
        const double h = 1.E-20;
        for (int i = 0; i < ns; ++i) {
          for (int j = 0; j < n; ++j) {
            yy_cplx_[j] = cplx(NV_Ith_S(y, j), h * NV_Ith_S(ys[i], j));
          }

          /* if y0 is the only parameter, use tangent linear
           * model(TLM). Otherwise use full forward sensitivity model.
           */
          if (B::is_var_y0) {
            fval_cplx_ =
              B::f_(t, yy_cplx_, B::theta_dbl_, B::x_r_, B::x_i_, B::msgs_);
          } else if (B::is_var_par) {
            std::transform(B::theta_dbl_.begin(),
                           B::theta_dbl_.end(),
                           theta_cplx_.begin(),
                           [](double r) -> cplx {return cplx(r, 0.0); });
            theta_cplx_[i] += cplx(0.0, h);
            fval_cplx_ =
              B::f_(t, yy_cplx_, theta_cplx_, B::x_r_, B::x_i_, B::msgs_);
          }

          std::transform(fval_cplx_.begin(),
                         fval_cplx_.end(),
                         B::fval_.begin(),
                         [&h](cplx x) -> double { return std::imag(x)/h; });
          for (int j = 0; j < n; ++j) NV_Ith_S(ysdot[i], j) = B::fval_[j];
        }
      }

      /**
       * return a lambda for sensitivity residual callback.
       * Here we use CSDA to do directional derivative
       * df/dy*s. This is efficient when the nb. of parameters
       * is less than the size of ODEs. But since we are doing
       * forward sensitivity, this is already assumed to be true.
       */
      static CVSensRhsFn sens_rhs() {
        return cvodes_sens_rhs<pk_cvodes_fwd_system<F, Tts, Ty0, Tpar, Lmm> >();
      }
    };

  }  // namespace dsolve
}  // namespace torsten

#endif
