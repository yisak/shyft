from shyft import api
import unittest


class GridPP(unittest.TestCase):
    """Verify GridPP API to process forecasts from met.no before feeding into shyft.
       Test diverse methods for transforming data sets to grids and vice versa.
       Expose API for IDW and BK from shyft core.
       Calculate bias timeseries using a Kalman filter algorithm. 
     """

    def setUp(self):
        self.cal = api.Calendar()
        self.dt = api.deltahours(1)
        self.nt = 24*10
        self.t0 = self.cal.time(2016, 1, 1)
        self.ta = api.Timeaxis2(self.t0, self.dt, self.nt)
        self.ta1 = api.Timeaxis(self.t0, self.dt, self.nt)

        self.geo_points = api.GeoPointVector()
        self.geo_points.append(api.GeoPoint( 100,  100, 1000))
        self.geo_points.append(api.GeoPoint(5100,  100, 1150))
        self.geo_points.append(api.GeoPoint( 100, 5100,  850))
        

    def _create_obs_set(self, geo_points):
        obs_set = api.TemperatureSourceVector()
        fx = lambda z : [15 for x in range(self.nt)]
        ts = api.Timeseries(ta=self.ta, values=fx(self.ta), point_fx=api.point_interpretation_policy.POINT_AVERAGE_VALUE)
        for gp in geo_points:
            # Add only one TS per GP, but should be several
            geo_ts = api.TemperatureSource(gp, ts)
            obs_set.append(geo_ts)
        return obs_set

    
    def _make_fc_from_obs(self, obs_set, bias):
        fc_set = api.TemperatureSourceVector()
        bias_ts = api.Timeseries(self.ta, fill_value=bias)
        for obs in obs_set:
            geo_ts = api.TemperatureSource(obs.mid_point(), obs.ts + bias_ts)
            fc_set.append(geo_ts)
        return fc_set

    
    def _predict_bias(self, obs_set, fc_set):
        # Return a set of bias_ts per observation geo_point
        bias_set = api.TemperatureSourceVector()
        kf = api.KalmanFilter()
        kbp = api.KalmanBiasPredictor(kf)
        kta = api.Timeaxis2(self.t0, api.deltahours(3), 8)
        for obs in obs_set:
            kbp.update_with_forecast(fc_set, obs.ts, kta)
            pattern = api.KalmanState.get_x(kbp.state)
            # a_ts = api.Timeseries(pattern, api.deltahours(3), self.ta)  # can do using ct of Timeseries, or:
            b_ts = api.create_periodic_pattern_ts(pattern, api.deltahours(3), self.ta.time(0), self.ta)  # function
            bias_set.append(api.TemperatureSource(obs.mid_point(), b_ts))
        return bias_set


    def test_calc_bias_should_match_observations(self):
        # Workflow from C++: Should we do the same from Python?
	    # IDW transform observation from set to grid 10 x 10 km. Call it forecast grid
	    # Simulate forecast offset of -2 degC 
	    # IDW transform forecast from frid to set
	    # Calculate bias set = observation set - forecast set
	    # IDW transform bias from set to grid
	    # Add bias grid to forecast grid
	    # IDW transform corrected forecast from grid to set
	    # Compare forecast to observation set => differences should be close to null

        obs_set = self._create_obs_set(self.geo_points)
        const_bias = 2.0
        fc_set = self._make_fc_from_obs(obs_set, const_bias)
        bias_set = self._predict_bias(obs_set, fc_set)
        self.assertEqual(len(bias_set), len(obs_set))
        for bias in bias_set:
            for i in range(len(bias.ts)):
                self.assertLess(bias.ts.value(i) - const_bias, 0.2)


if __name__ == "__main__":
    unittest.main()
