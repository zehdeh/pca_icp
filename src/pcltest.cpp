#include "pcltest.h"

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/common/centroid.h>


int pclTest(const int argc, char** const argv) {
	if(argc == 1) {
		std::cout << "Please specify an .OBJ-file!" << std::endl;
		return -1;
	}
	pcl::PointCloud<pcl::PointNormal>::Ptr input(new pcl::PointCloud<pcl::PointNormal>);

	if(pcl::io::loadOBJFile(argv[1], *input) == -1) {
		PCL_ERROR("Couldn't read file!\n");
		return -1;
	}

	pcl::ConditionAnd<pcl::PointNormal>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointNormal>());
	range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
		new pcl::FieldComparison<pcl::PointNormal>("y", pcl::ComparisonOps::GT, 650.0))
	);
	pcl::ConditionalRemoval<pcl::PointNormal> condrem;
	condrem.setCondition(range_cond);
	condrem.setInputCloud(input);
	condrem.setKeepOrganized(true);
	condrem.filter(*input);

	pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond2(new pcl::ConditionOr<pcl::PointNormal>());
	range_cond2->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
		new pcl::FieldComparison<pcl::PointNormal>("normal_x", pcl::ComparisonOps::GT, 0.5))
	);
	range_cond2->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
		new pcl::FieldComparison<pcl::PointNormal>("normal_x", pcl::ComparisonOps::LT, -0.5))
	);
	condrem.setCondition(range_cond2);
	condrem.setInputCloud(input);
	condrem.setKeepOrganized(true);
	condrem.filter(*input);
	

	Eigen::Vector4f centroid;
	pcl::PointCloud<pcl::PointNormal>::Ptr demeaned(new pcl::PointCloud<pcl::PointNormal>);
	pcl::compute3DCentroid(*input, centroid);
	pcl::demeanPointCloud(*input, centroid, *demeaned);

	std::cout << pcl::getFieldsList<pcl::PointNormal>(*demeaned) << std::endl;
	//pcl::PointCloud<pcl::PointXYZ>

	pcl::visualization::PCLVisualizer viewer("Simple Cloud Viewer");
	viewer.addPointCloud<pcl::PointNormal>(demeaned, "sample cloud");
	viewer.addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(demeaned,demeaned,1,2.0f);
	viewer.addCoordinateSystem(1.0);
	viewer.initCameraParameters();

	while(!viewer.wasStopped()) {
		viewer.spinOnce(100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
	
	return 0;
}
