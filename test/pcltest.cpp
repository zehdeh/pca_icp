#include "pcltest.h"

#include <vector>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/centroid.h>

void mouseEventOccurred(const pcl::visualization::MouseEvent &event, void* viewer_void) {
	if(event.getButton() == pcl::visualization::MouseEvent::RightButton && event.getType() == pcl::visualization::MouseEvent::MouseButtonRelease) {
		std::cout << "Right mouse button released!" << std::endl;
	}
}


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

	// Conditional removal: everything y < 650
	pcl::ConditionAnd<pcl::PointNormal>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointNormal>());
	range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
		new pcl::FieldComparison<pcl::PointNormal>("y", pcl::ComparisonOps::GT, 650.0))
	);
	pcl::ConditionalRemoval<pcl::PointNormal> condrem;
	condrem.setCondition(range_cond);
	condrem.setInputCloud(input);
	condrem.setKeepOrganized(true);
	condrem.filter(*input);

	// Conditional removal: every normal -0.5 < n_x < 0.5
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
	
	// Remove statistical outliers: neighbour distance > 2.0
	pcl::StatisticalOutlierRemoval<pcl::PointNormal> sor;
	sor.setInputCloud(input);
	sor.setMeanK(50);
	sor.setStddevMulThresh(2.0);
	sor.filter(*input);

	// Center cloud
	Eigen::Vector4f centroid;
	pcl::PointCloud<pcl::PointNormal>::Ptr demeaned(new pcl::PointCloud<pcl::PointNormal>);
	pcl::compute3DCentroid(*input, centroid);
	pcl::demeanPointCloud(*input, centroid, *demeaned);

	//pcl::PointCloud<pcl::PointXYZ>

	float meanXLeft = 0;
	int numXLeft = 0;
	float meanXRight = 0;
	int numXRight = 0;

	std::vector<float> pointsXLeft;
	std::vector<float> pointsXRight;
	for(size_t i = 0; i < demeaned->points.size(); i++) {
		if(demeaned->points[i].x > 1.0) {
			meanXLeft += demeaned->points[i].x;
			pointsXLeft.push_back(demeaned->points[i].x);
			numXLeft++;
		}
		if(demeaned->points[i].x < -1.0) {
			meanXRight += demeaned->points[i].x;
			pointsXRight.push_back(demeaned->points[i].x);
			numXRight++;
		}
	}
	meanXLeft = (meanXLeft / numXLeft);
	meanXRight = (meanXRight / numXRight);

	for(float& f : pointsXLeft) {
		f -= meanXLeft;
	}
	for(float& f : pointsXRight) {
		f -= meanXRight;
	}

	float stdDeviationXLeft = 0;
	float stdDeviationXRight = 0;
	float planeMeanXLeft = 0;
	float planeMeanXRight = 0;
	for(float& f : pointsXLeft) {
		stdDeviationXLeft += pow(f,2);
		planeMeanXLeft += f;
	}
	stdDeviationXLeft = sqrt(stdDeviationXLeft / (pointsXLeft.size() - 1));
	planeMeanXLeft = planeMeanXLeft / pointsXLeft.size();
	
	for(float& f : pointsXRight) {
		stdDeviationXRight += pow(f,2);
		planeMeanXRight += f;
	}
	//stdDeviationXRight = sqrt((1/(pointsXRight.size() - 1))*stdDeviationXRight);
	stdDeviationXRight = sqrt(stdDeviationXRight / (pointsXRight.size() - 1));
	planeMeanXRight = planeMeanXRight / pointsXRight.size();


	std::cout << "Mean left: " << meanXLeft << std::endl;
	std::cout << "Mean right: " << meanXRight << std::endl;
	std::cout << "Mean distance: " << (meanXLeft - meanXRight) << std::endl;
	std::cout << std::endl;
	std::cout << "Std deviation left plane: " << stdDeviationXLeft << std::endl;
	std::cout << "Std deviation right plane: " << stdDeviationXRight << std::endl;
	std::cout << "Mean left plane: " << planeMeanXLeft << std::endl;
	std::cout << "Mean right plane: " << planeMeanXLeft << std::endl;

	pcl::visualization::PCLVisualizer viewer("Simple Cloud Viewer");
	viewer.addPointCloud<pcl::PointNormal>(demeaned, "sample cloud");
	viewer.addPointCloudNormals<pcl::PointNormal, pcl::PointNormal>(demeaned,demeaned,1,2.0f);
	viewer.addCoordinateSystem(1.0);
	viewer.initCameraParameters();
	viewer.registerMouseCallback(mouseEventOccurred, (void*)&viewer);

	while(!viewer.wasStopped()) {
		viewer.spinOnce(100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
	
	return 0;
}
