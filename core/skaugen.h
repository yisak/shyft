#pragma once
///	Copyright 2012 Statkraft Energi A/S
///
///	This file is part of SHyFT.
///
///	SHyFT is free software: you can redistribute it and/or modify it under the terms of
/// the GNU Lesser General Public License as published by the Free Software Foundation,
/// either version 3 of the License, or (at your option) any later version.
///
///	SHyFT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
/// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
/// PURPOSE. See the GNU Lesser General Public License for more details.
///
///	You should have received a copy of the GNU Lesser General Public License along with
/// SHyFT, usually located under the SHyFT root directory in two files named COPYING.txt
/// and COPYING_LESSER.txt.	If not, see <http://www.gnu.org/licenses/>.
///
/// This implementation is a slightly improved and ported version of Skaugen's snow routine 
/// implemented in R, see [ref].

#include <math.h>
#include <iomanip>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/minima.hpp>
#include "utctime_utilities.h"
#include "timeseries.h"

namespace shyft {
    namespace core {
        namespace skaugen {

            class mass_balance_error : public std::exception {};

            typedef boost::math::policies::policy<boost::math::policies::digits10<16> > acc_policy;
            typedef boost::math::gamma_distribution<double, acc_policy> gamma_dist;

            struct statistics {
                const double alpha_0;
                const double d_range;
                const double unit_size;

                statistics(double alpha_0, double d_range, double unit_size)
                  : alpha_0(alpha_0), d_range(d_range), unit_size(unit_size) { /* Do nothing */ }

                static inline double c(long unsigned int n, double d_range) {
                    return exp(-(double)n/d_range);
                }

                static inline double sca_rel_red(long unsigned int u, long unsigned int n, double unit_size, double nu_a, double alpha) {
                    const double nu_m = ((double)u/n)*nu_a;
                    // Note; We use m (melt) in stead of s (smelt) due to language conversion
                    // Compute: Find X such that f_a(X) = f_m(X)
                    const gamma_dist g_m(nu_m, 1.0/alpha);
                    const gamma_dist g_a(nu_a, 1.0/alpha);
                    const double g_a_mean = boost::math::mean(g_a);
                    auto zero_func = [&] (const double& x) { 
                        return boost::math::pdf(g_m, x) - boost::math::pdf(g_a, x); } ;

                    double lower = boost::math::mean(g_m);
                    double upper = boost::math::tools::brent_find_minima(zero_func, 0.0, g_a_mean, 2).first;  // TODO: Is this robust enough??
                    while (boost::math::pdf(g_m, lower) < boost::math::pdf(g_a, lower))
                        lower *= 0.9;

                    boost::uintmax_t max_iter = 100;
                    boost::math::tools::eps_tolerance<double> tol(10); // 10 bit precition on result
                    typedef std::pair<double, double> result_t;
                    result_t res = boost::math::tools::bisect(zero_func, lower, upper, tol, max_iter);
                    //result_t res = boost::math::tools::toms748_solve(zero_func, lower, upper, tol, max_iter); // TODO: Boost bug causes this to crash!!
                    const double x = (res.first + res.second)*0.5; // TODO: Check that we converged
                    // Compute: {m,a} = \int_0^x f_{m,a} dx
                    const double m = boost::math::cdf(g_m, x);
                    const double a = boost::math::cdf(g_a, x);
                    return a + 1.0 - m;
                }

                double c(long unsigned int n) const { return c(n, d_range); }

                double sca_rel_red(long unsigned int u, long unsigned int n, double nu_a, double alpha) const {
                    return sca_rel_red(u, n, unit_size, nu_a, alpha);
                }
            };

            struct parameter {
                double alpha_0;
                double d_range;
                double unit_size;
                double max_water_fraction;
                double tx;
                double cx;
                double ts;
                double cfr;
                parameter() {}

                parameter(double alpha_0, double d_range, double unit_size, 
                          double max_water_fraction, double tx, double cx, double ts, double cfr)
                 : alpha_0(alpha_0), d_range(d_range),
                   unit_size(unit_size), max_water_fraction(max_water_fraction),
                   tx(tx), cx(cx), ts(ts), cfr(cfr) { /* Do nothing */ }
            };


            struct state {
                double nu;
                double alpha;
                double sca;
                double swe;
                double free_water;
                double residual;
                unsigned long num_units;
                state(double nu=4.077, double alpha=40.77, double sca=0.0,
                      double swe=0.0, double free_water=0.0, double residual=0.0, unsigned long num_units=0)
                 : nu(nu), alpha(alpha), sca(sca), swe(swe), free_water(free_water),
                   residual(residual), num_units(num_units) {}
            };


            struct response {
                double outflow = 0.0;
                double total_stored_water = 0.0;
            };

            template<class P, class S, class R> //, class PM> // TODO: Implement a physical model for the gamma snow swe/sca dynamics
            class calculator {
              private:
                const double snow_tol = 1.0e-10;
                //PM pm; // TODO: Implement a physical model for the gamma snow swe/sca dynamics
              public:
                //Skaugen(PM& pm) : pm(pm) { /* Do nothing */ } // TODO Implement a physical model for the gamma snow swe/sca dynamics
                calculator() { /* Do nothing */ }
                void step(shyft::timeseries::utctimespan dt,
                          const P& p,
                          const double T,
                          const double prec,
                          const double rad,
                          const double wind_speed,
                          S& s,
                          R& r) const {
                    const double unit_size = p.unit_size;

                    // Redistribute residual, if possible:
                    double corr_prec = prec;
                    if (s.residual > 0.0) {
                        corr_prec += s.residual;
                        s.residual = 0.0;
                    } else if (prec > 0.0) {
                        if (prec + s.residual > 0.0) {
                            corr_prec += s.residual;
                            s.residual = 0.0;
                        } else {
                            s.residual += prec;
                            corr_prec = 0.0;
                        }
                    }

                    // Perhaps Eli can come up with the physics. First hard coded degree day model here
                    const double step_in_days = dt/86400.0;
                    const double snow = T < p.tx ? corr_prec : 0.0;
                    const double rain = T < p.tx ? 0.0 : corr_prec;

                    //PM::calculate_melt(T, prec, rad, wind_speed, snow, rain, pot_melt); // TODO: Resolve interface to physical model

                    if (s.sca*s.swe < p.unit_size && snow < snow_tol) {
                        // Set state and response and return
                        r.outflow = rain + s.sca*(s.swe + s.free_water) + s.residual;
                        r.total_stored_water = 0.0;
                        s.residual = 0.0; 
                        if (r.outflow < 0.0) {
                            s.residual = r.outflow;
                            r.outflow = 0.0;
                        }
                        s.nu = p.alpha_0*p.unit_size;
                        s.alpha = p.alpha_0;
                        s.sca = 0.0;
                        s.swe = 0.0;
                        s.free_water = 0.0;
                        s.num_units = 0;
                        return;
                    }

                    const double alpha_0 = p.alpha_0;
                    double swe = s.swe; // Conditional value!
                    const double sca_old = s.sca;

                    long unsigned int nnn = s.num_units;
                    double sca = sca_old;
                    double nu = s.nu;
                    double alpha = s.alpha;

                    if (nnn > 0)
                        nu *= nnn; // Scaling nu with the number of accumulated units internally. Back scaled before returned.
                    else {
                        nu = alpha_0*p.unit_size;
                        alpha = alpha_0;
                    }

                    double total_new_snow = snow;
                    double lwc = s.free_water;  // Conditional value
                    const double total_storage = swe + lwc; // Conditional value

                    double pot_melt = p.cx*step_in_days*(T - p.ts);
                    if (pot_melt < 0.0) { // Refreeze
                        pot_melt *= p.cfr;
                        if (pot_melt + lwc < 0.0) {
                            pot_melt = -lwc;
                        }
                        total_new_snow -= sca*pot_melt;  // New snow and refreeze of old free water
                        lwc += pot_melt;  // EA and OS: We move water from lwc to total new snow, so both add and subtract
                        //lwc += sca*pot_melt;  // EA and OS: We move water from lwc to total new snow, so both add and subtract
                        pot_melt = 0.0;  // EA and OS: Water has been moved, so zero out the refreeze variable
                    } else {
                        if (pot_melt > total_new_snow) {
                            pot_melt -= total_new_snow;
                            total_new_snow = 0.0;
                        } else {
                            total_new_snow -= pot_melt;
                            pot_melt = 0.0;
                        }
                    }

                    statistics stat(alpha_0, p.d_range, unit_size);

                    long unsigned int n = 0;
                    long unsigned int u = 0;

                    // Accumulation
                    if (total_new_snow > 0.1) {
                        n = lrint(total_new_snow/unit_size);
                        if (n == 0) n = 1;
                        if (nnn == 0) { // First snowfall, simple case
                            sca = 1.0;
                            if (n == 1) {
                                alpha = alpha_0;
                                nu = alpha_0*unit_size;
                            } else
                                compute_shape_vars(stat, nnn, n, 0, sca, 0.0, alpha, nu);
                            swe = n*unit_size;
                        } else {
                            compute_shape_vars(stat, nnn, n, 0, sca, 0.0, alpha, nu);
                            nnn = lrint(nnn*sca) + n;
                            swe = nnn*unit_size;
                            sca = 1.0;
                        }
                        nnn = lrint(swe/unit_size);  // Update after accumulation
                    }

                    // Melting // Eli thinks there might possibly be something not completely right here ...
                    if (pot_melt > 0.1) {
                        u = lrint(pot_melt/unit_size);
                        if (nnn < u + 2) {
                            nnn = 0;
                            alpha = alpha_0;
                            nu = alpha_0*unit_size;
                            swe = 0.0;
                            lwc = 0.0;
                            sca = 0.0;
                        } else {
                            const double rel_red_sca = stat.sca_rel_red(u, nnn, nu, alpha);
                            const double sca_scale_factor = 1.0 - rel_red_sca;
                            sca = sca_old*sca_scale_factor;
                            swe = (nnn - u)/sca_scale_factor*unit_size;

                            if (swe >= nnn*unit_size) {
                                u = nnn*rel_red_sca + 1;
                                swe = (nnn - u)/sca_scale_factor*unit_size;
                                if (swe > 0.0)
                                    compute_shape_vars(stat, nnn, n, u, sca, rel_red_sca, alpha, nu);
                                else
                                    sca = 0.0;
                            }

                            if (sca < 0.005) {
                                nnn = 0;
                                alpha = alpha_0;
                                nu = alpha_0*unit_size;
                                swe = 0.0;
                                lwc = 0.0;
                                sca = 0.0;
                            } else {
                                //compute_shape_vars(stat, nnn, n, u, sca, rel_red_sca, alpha, nu);
                                nnn = lrint(swe/unit_size);
                                swe = nnn*unit_size;
                            }
                        }
                    }

                    // Eli and Ola, TODO: 
                    // This section is quite hackish, but makes sure that we have control on the local mass balance that is constantly
                    // violated due to discrete melting/accumulation events. A trained person should rethink the logic and have a goal of
                    // minimizing the number of if-branches below. Its embarrasingly many of them...
                    // The good news is that the residual is in the interval [-0.1, 0.1] for the cases we've investigated, and that 15 
                    // years of simulation on real data gives a O(1.0e-11) accumulated mass balance violation (in mm).

                    if (sca_old*s.swe > sca*swe) { // (Unconditional) melt 
                        lwc += std::max(0.0, s.swe - swe); // Add melted water to free water in snow
                    }
                    lwc *= std::min(1.0, sca_old/sca); // Scale lwc (preserve total free water when sca increases)
                    lwc = std::min(lwc, swe*p.max_water_fraction); // Max is max, you know ;)
                    double discharge = sca_old*total_storage + snow - sca*(swe + lwc); // We consider rain later

                    // If discharge is negative, recalculate new lwc to take the slack
                    if (discharge < 0.0) {
                        lwc = (sca_old*total_storage + snow)/sca - swe;
                        discharge = 0.0;
                        // Not enough water in the snow for the negative outflow
                        if (lwc < 0.0) {
                            lwc = 0.0;
                            discharge = sca_old*total_storage + snow - sca*swe;
                            // Giving up and adding to residual. TODO: SHOULD BE CHECKED, and preferably REWRITTEN!!
                            if (discharge < 0.0) {
                                s.residual += discharge;
                                discharge = 0.0;
                            }
                        }
                    }

                    // If no runoff, discharge is numerical noise and should go into residual.
                    if (snow > 0.0 && rain == 0.0 && pot_melt <= 0.0) {
                        s.residual += discharge;
                        discharge = 0.0;
                    }

                    // Add rain
                    if (rain > swe*p.max_water_fraction - lwc) {
                        discharge += sca*(rain - (swe*p.max_water_fraction - lwc)) + rain*(1.0 - sca); // exess water in snowpack + uncovered
                        lwc = swe*p.max_water_fraction; // No more room for water
                    } else {
                        lwc += rain; // Rain becomes water in the snow
                        discharge += rain*(1.0 - sca); // Only uncovered area gives rain discharge
                    }

                    // Negative discharge may come from roundoff due to discrete model, add to residual and get rid of it later
                    if (discharge <= 0.0) { 
                        s.residual += discharge;
                        discharge = 0.0;
                    } else { // Discharge is positive
                        // We have runoff, so get rid of as much residual as possible (without mass balance violation)
                        if (discharge >= -s.residual) { 
                            discharge += s.residual;
                            s.residual = 0.0;
                        // More residual than discharge - get rid of as much as possible and zero out the runoff
                        } else {
                            s.residual += discharge;
                            discharge = 0.0;
                        }
                    }

                    if (nnn > 0)
                        nu /= nnn;

                    r.outflow = discharge;
                    r.total_stored_water = sca*(swe + lwc);
                    s.nu = nu;
                    s.alpha = alpha;
                    s.sca = sca;
                    s.swe = swe;
                    s.free_water = lwc;
                    s.num_units = nnn;
                }

                static inline void compute_shape_vars(const statistics& stat,
                                                      long unsigned int nnn,
                                                      long unsigned int n,
                                                      long unsigned int u,
                                                      double sca,
                                                      double rel_red_sca,
                                                      double& alpha,
                                                      double& nu) {
                    const double alpha_0 = stat.alpha_0;
                    const double nu_0 = stat.alpha_0*stat.unit_size;
                    const double dyn_var = nu/(alpha*alpha);
                    const double init_var = nu_0/(alpha_0*alpha_0);
                    double tot_var = 0.0;
                    double tot_mean = 0.0;
                    if (n > 0) { // Accumulation
                        if (nnn == 0) { // No previous snow pack
                            tot_var = n*init_var*(1 + (n - 1)*stat.c(n));
                            tot_mean = n*nu_0/alpha_0;
                        } else  {
                            const double old_var_cov = (nnn + n)*init_var*(1 + ((nnn + n) - 1)*stat.c(nnn + n));
                            const double new_var_cov = n*init_var*(1 + (n - 1)*stat.c(n));
                            tot_var = old_var_cov*sca*sca + new_var_cov*(1.0 - sca)*(1.0 - sca);
                            tot_mean = (sca*(nnn + n) + (1.0 - sca)*n)*stat.unit_size;
                        }
                    }
                    if (u > 0) { // Ablation
                        const double factor = (dyn_var/(nnn*init_var) + 1.0 + (nnn - 1)*stat.c(nnn))/(2*nnn);
                        const double non_cond_mean = (nnn - u)*stat.unit_size;
                        tot_mean = non_cond_mean/(1.0 - rel_red_sca);
                        const long unsigned int cond_u = lrint((1.0 - rel_red_sca)*nnn - (nnn - u));
                        const double auto_var  = cond_u > 0 ? init_var*cond_u*(1.0 + (cond_u - 1.0)*stat.c(cond_u)) : 0.0;
                        const double cross_var = cond_u > 0 ? init_var*cond_u*2.0*factor*cond_u : 0.0;
                        tot_var = dyn_var + auto_var - cross_var;
                    }

                    if (fabs(tot_mean) < 1.0e-7) {
                        nu = nu_0;
                        alpha = alpha_0;
                        return;
                    } 
                    nu = tot_mean*tot_mean/tot_var;
                    alpha = nu/(stat.unit_size*lrint(tot_mean/stat.unit_size));
                }
            };
       } // skaugen
    } // core
} // shyft
