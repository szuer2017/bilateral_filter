#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/median_filter.h>
#include <pcl/filters/convolution_3d.h>
#include <iostream>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/property_map.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/IO/write_xyz_points.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/remove_outliers.h>
#include <CGAL/bilateral_smooth_point_set.h>
#include <CGAL/edge_aware_upsample_point_set.h>
#include <CGAL/tags.h>
#include <utility> // defines std::pair

using namespace std;
using namespace pcl;

typedef pcl::PointXYZ PointT;
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3  Vector;

typedef std::pair<Point, Vector> PointVectorPair;

#ifdef CGAL_LINKED_WITH_TBB
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

int main(int argc, char* argv[])
{
	
	//read the pointcloud
	pcl::PointCloud<PointT>::Ptr src(new pcl::PointCloud<PointT>);

	pcl::PointCloud<PointT>::Ptr cloud_RF(new pcl::PointCloud<PointT>);

	pcl::PointCloud<PointT>::Ptr cloud_rad(new pcl::PointCloud<PointT>);

	pcl::io::loadPCDFile("3d_data.pcd", *src);

	//radius filter
	pcl::RadiusOutlierRemoval<PointT> rad;
	rad.setInputCloud(src);
	rad.setRadiusSearch(0.1);
	rad.setMinNeighborsInRadius(30);
	rad.filter(*cloud_rad);

	//stastic filter
	pcl::VoxelGrid<PointT> voxelgrid;
	voxelgrid.setInputCloud(cloud_rad);
	voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
	voxelgrid.filter(*cloud_RF);


	//compute the normal
	pcl::PointCloud<pcl::Normal>::Ptr pcNormal(new pcl::PointCloud<pcl::Normal>);

	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	tree->setInputCloud(cloud_RF);

	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	ne.setInputCloud(cloud_RF);
	ne.setSearchMethod(tree);
	ne.setKSearch(25);

	ne.compute(*pcNormal);

	//connection the points and normal
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normal(new pcl::PointCloud<pcl::PointXYZINormal>);
	pcl::concatenateFields(*cloud_RF, *pcNormal, *cloud_with_normal);

	//save  File  format as  XYZ
	int size_t = cloud_with_normal->size();

	ofstream file("hybrid_filter_test.xyz");
	file.precision(6);
	for (int i = 0; i < size_t;i++)
	{
		std::string error_p = "1.#QNAN0";//1.#QNAN0  -1.#IND00
		std::string x = to_string(cloud_with_normal->points[i].x);
		std::string y = to_string(cloud_with_normal->points[i].y);
		std::string z = to_string(cloud_with_normal->points[i].z);
		std::string n_x = to_string(cloud_with_normal->points[i].normal_x);
		std::string n_y = to_string(cloud_with_normal->points[i].normal_y);
		std::string n_z = to_string(cloud_with_normal->points[i].normal_z);
		if (x==error_p||n_x == error_p || n_y == error_p || n_y == error_p)
		{
			x=y=z=n_x  =  n_y  =  n_z  = "0.000000";
		}
		file <<x<< " "
			<< y << " "
			<< z << " "
			<< n_x << " " << n_y << " " << n_z << endl;
	}

	file.close();

	//prossessor with Bilateral Filter in the CGAL library

	const char* input_filename = (argc > 1) ? argv[1] : "hybrid_filter_test.xyz";
	const char* output_filename = (argc > 2) ? argv[2] : "hybrid_ear1.xyz";

	std::vector<PointVectorPair> points;
	std::ifstream stream(input_filename);
	if (!stream ||
		!CGAL::read_xyz_points(stream,
		std::back_inserter(points),
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
		normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>())))
	{
		std::cerr << "Error: cannot read file " << input_filename << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << "read the all data!"<<std::endl;
	// Algorithm parameters
	int k = 100;                 // size of neighborhood. The bigger the smoother the result will be.
	// This value should bigger than 1.
	double sharpness_angle = 25; // control sharpness of the result.
	// The bigger the smoother the result will be
	int iter_number = 3;         // number of times the projection is applied


	for (int i = 0; i < iter_number; ++i)
	{
		/* double error = */
		
		CGAL::bilateral_smooth_point_set <Concurrency_tag>(
			points,
			k,
			CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
			normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()).
			sharpness_angle(sharpness_angle));
	}

	std::cout << "bilateral filter have done" << endl;

	//perform the EAR 
	const std::size_t  Num_out = points.size() * 3;

	CGAL::edge_aware_upsample_point_set<Concurrency_tag>(
		points,
		std::back_inserter(points),  
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
		normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()).
		sharpness_angle(25).
		edge_sensitivity(0).
		neighbor_radius(0.25).
		number_of_output_points(Num_out)
		);

	std::cout << "EAR Caculation Done !" << endl;

	std::ofstream out(output_filename);
	if (!out ||
		!CGAL::write_xyz_points(
		out, points,
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
		normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>())))
	{
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;








}