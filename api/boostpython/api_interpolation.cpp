#include "boostpython_pch.h"
#include "api/api.h"
#include "core/inverse_distance.h"
#include "core/bayesian_kriging.h"
#include "core/region_model.h"

namespace expose {
    using namespace boost::python;
    namespace sa = shyft::api;
    namespace sc = shyft::core;
	namespace sta = shyft::time_axis;
	namespace btk = shyft::core::bayesian_kriging;
	namespace idw = shyft::core::inverse_distance;

	typedef std::vector<sc::geo_point> geo_point_vector;
	typedef std::vector<sa::TemperatureSource> geo_temperature_vector;
	typedef std::shared_ptr<geo_temperature_vector> geo_temperature_vector_;
	typedef std::vector<sa::PrecipitationSource> geo_precipitation_vector;
	typedef std::shared_ptr<geo_precipitation_vector> geo_precipitation_vector_;

	template <typename VectorT>
	static std::shared_ptr<VectorT> make_dest_geo_ts(const geo_point_vector& points, sta::fixed_dt time_axis) {
		auto dst = std::make_shared<VectorT>();
		dst->reserve(points.size());
		double std_nan = std::numeric_limits<double>::quiet_NaN();
		for (const auto& gp : points)
			dst->emplace_back(gp, sa::apoint_ts(time_axis, std_nan, shyft::timeseries::point_interpretation_policy::POINT_AVERAGE_VALUE));
		return dst;
	}

	template <typename VectorT>
	static void validate_parameters(const std::shared_ptr<VectorT> & src, const geo_point_vector& dst_points, sta::fixed_dt time_axis) {
		if (src==nullptr || src->size()==0 || dst_points.size()==0)
			throw std::runtime_error("the supplied src and dst_points should be non-null and have at least one time-series");
		if (time_axis.size()==0 || time_axis.delta()==0)
			throw std::runtime_error("the supplied destination time-axis should have more than 0 element, and a delta-t larger than 0");
	}

    ///< a local wrapper with api-typical checks on the input to support use from python
    static geo_temperature_vector_ bayesian_kriging_temperature(geo_temperature_vector_ src,const geo_point_vector& dst_points,shyft::time_axis::fixed_dt time_axis,btk::parameter btk_parameter) {
        using namespace std;
        typedef shyft::timeseries::average_accessor<typename shyft::api::apoint_ts, shyft::time_axis::fixed_dt> btk_tsa_t;
        // 1. some minor checks to give the python user early warnings.
        validate_parameters(src, dst_points, time_axis);
        auto dst = make_dest_geo_ts<geo_temperature_vector>(dst_points, time_axis);
        // 2. then run btk to fill inn the results
        if(src->size()>1) {
            btk::btk_interpolation<btk_tsa_t>(begin(*src), end(*src), begin(*dst), end(*dst),time_axis, btk_parameter);
        } else {
            // just one temperature ts. just a a clean copy to destinations
            btk_tsa_t tsa((*src)[0].ts, time_axis);
            sa::apoint_ts temp_ts(time_axis, 0.0);
            for(size_t i=0;i<time_axis.size();++i) temp_ts.set(i, tsa.value(i));
            for(auto& d:*dst) d.ts=temp_ts;
        }
        return dst;
    }

    static void btk_interpolation() {
        typedef shyft::core::bayesian_kriging::parameter BTKParameter;

        class_<BTKParameter>("BTKParameter","BTKParameter class with time varying gradient based on day no")
            .def(init<double,double>(args("temperature_gradient","temperature_gradient_sd"),"specifying default temp.grad(not used) and std.dev[C/100m]"))
            .def(init<double,double,double,double,double,double>(args("temperature_gradient","temperature_gradient_sd", "sill", "nugget", "range", "zscale"),"full specification of all parameters"))
            .def("temperature_gradient",&BTKParameter::temperature_gradient,args("p"),"return default temp.gradient based on day of year calculated for midst of utcperiod p")
            .def("temperature_gradient_sd",&BTKParameter::temperature_gradient_sd,"returns Prior standard deviation of temperature gradient in [C/m]" )
            .def("sill",&BTKParameter::sill,"Value of semivariogram at range default=25.0")
            .def("nug",&BTKParameter::nug,"Nugget magnitude,default=0.5")
            .def("range",&BTKParameter::range,"Point where semivariogram flattens out,default=200000.0")
            .def("zscale",&BTKParameter::zscale,"Height scale used during distance computations,default=20.0")
            ;
        def("bayesian_kriging_temperature",bayesian_kriging_temperature,
            "Runs kriging for temperature sources and project the temperatures out to the destination geo-timeseries\n"
            "\n\n\tNotice that bayesian kriging is currently not very efficient for large grid inputs,\n"
            "\tusing only one thread, and considering all source-timeseries (entire grid) for all destinations\n"
            "\tFor few sources, spread out on a grid, it's quite efficient should work well\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "src : TemperatureSourceVector\n"
            "\t input a geo-located list of temperature time-series with filled in values (some might be nan etc.)\n\n"
            "dst : GeoPointVector\n"
            "\tthe GeoPoints,(x,y,z) locations to interpolate into\n"
            "time_axis : Timeaxis, - the destination time-axis, recall that the inputs can be any-time-axis, \n"
            "\tand they are transformed and interpolated into the destination-timeaxis\n"
            "btk_parameter:BTKParameter\n"
            "\t the parameters to be used during interpolation\n\n"
            "Returns\n"
            "-------\n"
            "TemperatureSourveVector, -with filled in temperatures according to their position, the idw_parameters and time_axis\n"
            );
    }

	static geo_temperature_vector_ idw_temperature(geo_temperature_vector_ src, const geo_point_vector& dst_points, shyft::time_axis::fixed_dt ta, idw::temperature_parameter idw_temp_p) {
		typedef shyft::timeseries::average_accessor<sa::apoint_ts, sc::timeaxis_t> avg_tsa_t;
		typedef sc::idw_compliant_geo_point_ts<sa::TemperatureSource, avg_tsa_t, sc::timeaxis_t> idw_gts_t;
		typedef idw::temperature_model<idw_gts_t, sa::TemperatureSource, idw::temperature_parameter, sc::geo_point, idw::temperature_gradient_scale_computer> idw_temperature_model_t;

		validate_parameters(src, dst_points, ta);
        auto dst = make_dest_geo_ts<geo_temperature_vector>(dst_points, ta);
		idw::run_interpolation<idw_temperature_model_t, idw_gts_t>(ta, *src, idw_temp_p, *dst,
			[](auto& d, size_t ix, double value) { d.set_value(ix, value); });

		return dst;
	}

	static geo_precipitation_vector_ idw_precipitation(geo_precipitation_vector_ src, const geo_point_vector& dst_points, shyft::time_axis::fixed_dt ta, idw::precipitation_parameter idw_p) {
		typedef shyft::timeseries::average_accessor<sa::apoint_ts, sc::timeaxis_t> avg_tsa_t;
		typedef sc::idw_compliant_geo_point_ts<sa::PrecipitationSource, avg_tsa_t, sc::timeaxis_t> idw_gts_t;
		typedef idw::precipitation_model<idw_gts_t, sa::PrecipitationSource, idw::precipitation_parameter, sc::geo_point> idw_precipitation_model_t;

		validate_parameters(src, dst_points, ta);
		auto dst = make_dest_geo_ts<geo_precipitation_vector>(dst_points, ta);
		idw::run_interpolation<idw_precipitation_model_t, idw_gts_t>(ta, *src, idw_p, *dst,
			[](auto& d, size_t ix, double value) { d.set_value(ix, value); });

		return dst;
	}

    static void idw_interpolation() {
        typedef shyft::core::inverse_distance::parameter IDWParameter;

        class_<IDWParameter>("IDWParameter",
                    "IDWParameter is a simple place-holder for IDW parameters\n"
                    "The two most common:\n"
                    "  max_distance \n"
                    "  max_members \n"
                    "Is used for interpolation.\n"
                    "Additionally it keep distance measure-factor,\n"
                    "so that the IDW distance is computed as 1 over pow(euclid distance,distance_measure_factor)\n"
					"zscale is used to discriminate neighbors that are at different elevation than target point.\n"
			)
			.def(init<int,optional<double,double>>(args("max_members","max_distance","distance_measure_factor"),"create IDW from supplied parameters"))
            .def_readwrite("max_members",&IDWParameter::max_members,"maximum members|neighbors used to interpolate into a point,default=10")
            .def_readwrite("max_distance",&IDWParameter::max_distance,"[meter] only neighbours within max distance is used for each destination-cell,default= 200000.0")
			.def_readwrite("distance_measure_factor",&IDWParameter::distance_measure_factor,"IDW distance is computed as 1 over pow(euclid distance,distance_measure_factor), default=2.0")
			.def_readwrite("zscale",&IDWParameter::zscale,"Use to weight neighbors having same elevation, default=1.0")
            ;
		def("idw_temperature", idw_temperature,
			"Runs inverse distance interpolation to project temperature sources out to the destination geo-timeseries\n"
			"\n"
			"Parameters\n"
			"----------\n"
			"src : TemperatureSourceVector\n"
			"\t input a geo-located list of temperature time-series with filled in values (some might be nan etc.)\n\n"
			"dst : GeoPointVector\n"
			"\tthe GeoPoints,(x,y,z) locations to interpolate into\n"
			"time_axis : Timeaxis, - the destination time-axis, recall that the inputs can be any-time-axis, \n"
			"\tthey are transformed and interpolated into the destination-timeaxis\n"
			"idw_para : IDWTemperatureParameter\n"
			"\t the parameters to be used during interpolation\n\n"
			"Returns\n"
			"-------\n"
			"TemperatureSourveVector, -with filled in temperatures according to their position, the idw_parameters and time_axis\n"
		);
		def("idw_precipitation", idw_precipitation,
			"Runs inverse distance interpolation to project precipitation sources out to the destination geo-timeseries\n"
			"\n"
			"Parameters\n"
			"----------\n"
			"src : PrecipitationSourceVector\n"
			"\t input a geo-located list of precipitation time-series with filled in values (some might be nan etc.)\n\n"
			"dst : GeoPointVector\n"
			"\tthe GeoPoints,(x,y,z) locations to interpolate into\n"
			"time_axis : Timeaxis, - the destination time-axis, recall that the inputs can be any-time-axis, \n"
			"\tthey are transformed and interpolated into the destination-timeaxis\n"
			"idw_para : IDWPrecipitationParameter\n"
			"\t the parameters to be used during interpolation\n\n"
			"Returns\n"
			"-------\n"
			"PrecipitationSourveVector, -with filled in precipitations according to their position, the idw_parameters and time_axis\n"
		);
        typedef shyft::core::inverse_distance::temperature_parameter IDWTemperatureParameter;
        class_<IDWTemperatureParameter,bases<IDWParameter>> ("IDWTemperatureParameter",
                "For temperature inverse distance, also provide default temperature gradient to be used\n"
                "when the gradient can not be computed.\n"
                "note: if gradient_by_equation is set true, and number of points >3, the temperature gradient computer\n"
                "      will try to use the 4 closes points and determine the 3d gradient including the vertical gradient.\n"
                "      (in scenarios with constant gradients(vertical/horizontal), this is accurate) \n"
            )
            .def(init<double,optional<int,double,bool>>(args("default_gradient","max_members","max_distance","gradient_by_equation"),"construct IDW for temperature as specified by arguments"))
            .def_readwrite("default_temp_gradient",&IDWTemperatureParameter::default_temp_gradient,"[degC/m], default=-0.006")
            .def_readwrite("gradient_by_equation",&IDWTemperatureParameter::gradient_by_equation,"if true, gradient is computed using 4 closest neighbors, solving equations to find 3D temperature gradients.")
            ;

        typedef shyft::core::inverse_distance::precipitation_parameter IDWPrecipitationParameter;
        class_<IDWPrecipitationParameter,bases<IDWParameter>>("IDWPrecipitationParameter",
                    "For precipitation,the scaling model needs the scale_factor.\n"
                    "adjusted_precipitation = precipitation* (scale_factor)^(z-distance-in-meters/100.0)\n"
                    "Ref to IDWParameter for the other parameters\n"
            )
            .def(init<double,optional<int,double>>(args("scale_factor", "max_members","max_distance"),"create IDW from supplied parameters"))
            .def_readwrite("scale_factor",&IDWPrecipitationParameter::scale_factor," ref. formula for adjusted_precipitation,  default=1.02")
        ;
    }

	static void interpolation_parameter() {
        typedef shyft::core::interpolation_parameter InterpolationParameter;
        namespace idw = shyft::core::inverse_distance;
        namespace btk = shyft::core::bayesian_kriging;
        class_<InterpolationParameter>("InterpolationParameter",
                 "The InterpolationParameter keep parameters needed to perform the\n"
                 "interpolation steps, IDW,BTK etc\n"
                 "It is used as parameter  in the model.run_interpolation() method\n"
            )
            .def(init<const btk::parameter&,const idw::precipitation_parameter&,const idw::parameter&,const idw::parameter&,const idw::parameter&>(args("temperature","precipitation","wind_speed","radiation","rel_hum"),"using BTK for temperature"))
            .def(init<const idw::temperature_parameter&,const idw::precipitation_parameter&,const idw::parameter&,const idw::parameter&,const idw::parameter&>(args("temperature","precipitation","wind_speed","radiation","rel_hum"),"using smart IDW for temperature, typically grid inputs"))
            .def_readwrite("use_idw_for_temperature",&InterpolationParameter::use_idw_for_temperature,"if true, the IDW temperature is used instead of BTK, useful for grid-input scenarios")
            .def_readwrite("temperature",&InterpolationParameter::temperature,"BTK for temperature (in case .use_idw_for_temperature is false)")
            .def_readwrite("temperature_idw",&InterpolationParameter::temperature_idw,"IDW for temperature(in case .use_idw_for_temperature is true)")
            .def_readwrite("precipitation",&InterpolationParameter::precipitation,"IDW parameters for precipitation")
            .def_readwrite("wind_speed", &InterpolationParameter::wind_speed,"IDW parameters for wind_speed")
            .def_readwrite("radiation", &InterpolationParameter::radiation,"IDW parameters for radiation")
            .def_readwrite("rel_hum",&InterpolationParameter::rel_hum,"IDW parameters for relative humidity")
            ;
    }

	void interpolation() {
        idw_interpolation();
        btk_interpolation();
        interpolation_parameter();
    }
}
