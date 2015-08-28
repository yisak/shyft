#pragma once
#include "geo_point.h"
namespace shyft {
    namespace core {

        /** \brief LandTypeFractions are used to describe 'type of land'
         *   like glacier,lake,reservoir and forest
         *   It is a designed as a part of GeoCellData (could be nested, but we take python/swig into consideration)
         *   It's of course questionable, since we could have individual models for each
         *   type of land, - but current approach is to use a cell-area, and then describe
         *   fractional properties
         */
        struct land_type_fractions {
            land_type_fractions():glacier_(0),lake_(0),reservoir_(0),forest_(0){}
            double glacier() const {return glacier_;}
			double lake() const {return lake_;}   // not regulated, assume time-delay until discharge
			double reservoir()const{return reservoir_;}// regulated, assume zero time-delay to discharge
			double forest() const {return forest_;}
			double unspecified() const {return 1.0 - glacier_-lake_-reservoir_-forest_;}
            void set_fractions(double glacier,double lake,double reservoir,double forest) {
                if(glacier+lake+reservoir+forest > 1.0  || (glacier<0.0 ||lake<0.0 || reservoir <0.0 || forest< 0.0 ) )
                   throw std::invalid_argument("LandTypeFractions:: must be >=0.0 and sum <= 1.0");
                glacier_=glacier;lake_=lake;reservoir_=reservoir;forest_=forest;
            }
          private:
			double glacier_;
			double lake_;   // not regulated, assume time-delay until discharge
			double reservoir_;// regulated, assume zero time-delay to discharge
			double forest_;
        };
        const double default_radiation_slope_factor=0.9;

        /** \brief geo_cell_data represents common constant geo_cell properties across several possible models and cell assemblies.
         *  The idea is that most of our algorithms uses one or more of these properties,
         *  so we provide a common aspect that keeps this together.
         *  Currently it keep the
         *   - mid-point geo_point, (x,y,z) (default 0)
         *     the area in m^2, (default 1000 x 1000 m2)
         *     land_type_fractions (unspecified=1)
         *     catchment_id   def (-1)
         *     radiation_slope_factor def 0.9
		 */
		struct geo_cell_data {
		    static const int default_area_m2=1000000;
			geo_cell_data() :area_m2(default_area_m2),catchment_id_(-1),radiation_slope_factor_(default_radiation_slope_factor){}

			geo_cell_data(geo_point mid_point,double area=default_area_m2,
                int catchment_id = -1, double radiation_slope_factor=default_radiation_slope_factor,const land_type_fractions& land_type_fractions=land_type_fractions()):
				mid_point_(mid_point), area_m2(area), catchment_id_(catchment_id),radiation_slope_factor_(radiation_slope_factor),fractions(land_type_fractions)
			{}
			const geo_point& mid_point() const { return mid_point_; }
			size_t catchment_id() const { return catchment_id_; }
			void set_catchment_id(size_t catchmentid) {catchment_id_=catchmentid;}
			double radiation_slope_factor() const { return radiation_slope_factor_; }
			const land_type_fractions& land_type_fractions_info() const { return fractions; }
			void set_land_type_fractions(const land_type_fractions& ltf) { fractions = ltf; }
			double area() const { return area_m2; }
		  private:

			geo_point mid_point_; // midpoint
			double area_m2; //m2
			size_t catchment_id_;
			double radiation_slope_factor_;
			land_type_fractions fractions;
			// geo-type  parts, interesting for some/most response routines, sum fractions should be <=1.0
		};
    }
} // shyft
