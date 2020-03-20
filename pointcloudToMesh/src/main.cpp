/*********************************
           HEADERS
**********************************/

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/file_io.h>
#include <pcl/io/ply/ply_parser.h>
#include <pcl/io/ply/ply.h>

#include <pcl/point_types.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

#include <pcl/range_image/range_image.h>

#include <pcl/common/transforms.h>
#include <pcl/common/geometry.h>
#include <pcl/common/common.h>
#include <pcl/common/common_headers.h>

#include <pcl/ModelCoefficients.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/gasd.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/filters/crop_box.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/surface/poisson.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/simplification_remove_unused_vertices.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/convex_hull.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/algorithm.hpp>
#include <boost/thread/thread.hpp>

#include <iostream>
#include <fstream>
#include <string>

void printUsage(const char *progName) {
    std::cout << "\nUsage: " << progName << " <input cloud> <surface method> <normal estimation method> <output dir>"
              << std::endl;
    std::cout << "surface method: \n '1' for poisson \n '2' for gp3" << std::endl;
    std::cout << "normal estimation method: \n '1' for normal estimation \n '2' for mls normal estimation" << std::endl;
}

void create_mesh(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PolygonMesh &triangles, int setKSearch, int nThreads, int depth,
        float pointWeight, float samplePNode, float scale, int isoDivide, int degree) {

    /* ****Translated point cloud to origin**** */
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -centroid[0], -centroid[1], -centroid[2];

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTranslated(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*cloud, *cloudTranslated, transform);

    /* ****kdtree search and msl object**** */
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_for_points(new pcl::search::KdTree <pcl::PointXYZ>);
    kdtree_for_points->setInputCloud(cloudTranslated);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());

    // normal cal
    std::cout << "Using normal method estimation...";

    pcl::NormalEstimationOMP <pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);

    n.setInputCloud(cloudTranslated);
    n.setSearchMethod(kdtree_for_points);
    n.setKSearch(20); //It was 20
    n.compute(*normals);//Normals are estimated using standard method.

    //pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal> ());
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    std::cout << "normal estimated [OK]" << std::endl;
    // end normal estimation


    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr kdtree_for_normals(new pcl::search::KdTree <pcl::PointNormal>);
    kdtree_for_normals->setInputCloud(cloud_with_normals);

    std::cout << "Applying surface meshing...";


    std::cout << "Using surface method: poisson ..." << std::endl;

//    int nThreads = 8;
//    int setKsearch = 10;
//    int depth = 12;
//    float pointWeight = 1.0;
//    float samplePNode = 1.2;
//    float scale = 1.1;
//    int isoDivide = 8;
    bool confidence = false;
    bool outputPolygons = false;
    bool manifold = false;
    int solverDivide = 8;
    printf("setKsearch %d, depth %d, pointWeight %f, samplePNode %f, scale %f, isoDivide %d", setKsearch, depth,
           pointWeight, samplePNode, scale, isoDivide);

    pcl::Poisson <pcl::PointNormal> poisson;

    poisson.setDepth(depth);//9
    poisson.setInputCloud(cloud_with_normals);
    poisson.setPointWeight(pointWeight);//4
    poisson.setDegree(degree);
    poisson.setSamplesPerNode(samplePNode);//1.5
    poisson.setScale(scale);//1.1
    poisson.setIsoDivide(isoDivide);//8
    poisson.setConfidence(confidence);
    poisson.setOutputPolygons(outputPolygons);
    poisson.setManifold(manifold);
    poisson.setSolverDivide(solverDivide);//8
    poisson.reconstruct(triangles);

    //pcl::PolygonMesh mesh2;
    //poisson.reconstruct(mesh2);
    //pcl::surface::SimplificationRemoveUnusedVertices rem;
    //rem.simplify(mesh2,triangles);

    std::cout << "[OK]" << std::endl;


}

void vizualizeMesh(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PolygonMesh &mesh) {

    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("MAP3D MESH"));

    int PORT1 = 0;
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, PORT1);
    viewer->setBackgroundColor(0, 0, 0, PORT1);
    viewer->addText("ORIGINAL", 10, 10, "PORT1", PORT1);

    int PORT2 = 0;
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, PORT2);
    viewer->setBackgroundColor(0, 0, 0, PORT2);
    viewer->addText("MESH", 10, 10, "PORT2", PORT2);
    viewer->addPolygonMesh(mesh, "mesh", PORT2);

    viewer->addCoordinateSystem();
    pcl::PointXYZ p1, p2, p3;

    p1.getArray3fMap() << 1, 0, 0;
    p2.getArray3fMap() << 0, 1, 0;
    p3.getArray3fMap() << 0, 0.1, 1;

    viewer->addText3D("x", p1, 0.2, 1, 0, 0, "x_");
    viewer->addText3D("y", p2, 0.2, 0, 1, 0, "y_");
    viewer->addText3D("z", p3, 0.2, 0, 0, 1, "z_");

    if (cloud->points[0].r <= 0 and cloud->points[0].g <= 0 and cloud->points[0].b <= 0) {
        pcl::visualization::PointCloudColorHandlerCustom <pcl::PointXYZRGB> color_handler(cloud, 255, 255, 0);
        viewer->removeAllPointClouds(0);
        viewer->addPointCloud(cloud, color_handler, "original_cloud", PORT1);
    } else {
        viewer->addPointCloud(cloud, "original_cloud", PORT1);
    }

    viewer->initCameraParameters();
    viewer->resetCamera();

    std::cout << "Press [q] to exit!" << std::endl;
    while (!viewer->wasStopped()) {
        viewer->spin();
    }
}

/*
void cloudPointFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,pcl::PointCloud<pcl::PointXYZ>::Ptr& filterCloud){

  std::cout << "Filtering point cloud..." << std::endl;
  std::cout << "Point cloud before filter:" << cloud->points.size()<< std::endl;

  pcl::RadiusOutlierRemoval<pcl::PointXYZ> radius_outlier_removal;
  radius_outlier_removal.setInputCloud(cloud);
  radius_outlier_removal.setRadiusSearch(0.01);
  radius_outlier_removal.setMinNeighborsInRadius(1);
  radius_outlier_removal.filter(*filterCloud);

  std::cout << "Point cloud after filter:" << filterCloud->points.size() << std::endl;
}
*/

int main(int argc, char **argv) {

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PolygonMesh cl;
    std::vector<int> filenames;
    bool file_is_pcd = false;
    bool file_is_ply = false;
    bool file_is_txt = false;
    bool file_is_xyz = false;

    if (argc < 5 or argc > 5) {
        printUsage(argv[0]);
        return -1;
    }

    pcl::console::TicToc tt;
    pcl::console::print_highlight("Loading ");

    filenames = pcl::console::parse_file_extension_argument(argc, argv, ".ply");
    if (filenames.size() <= 0) {
        filenames = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
        if (filenames.size() <= 0) {
            filenames = pcl::console::parse_file_extension_argument(argc, argv, ".txt");
            if (filenames.size() <= 0) {
                filenames = pcl::console::parse_file_extension_argument(argc, argv, ".xyz");
                if (filenames.size() <= 0) {
                    printUsage(argv[0]);
                    return -1;
                } else if (filenames.size() == 1) {
                    file_is_xyz = true;
                }
            } else if (filenames.size() == 1) {
                file_is_txt = true;
            }
        } else if (filenames.size() == 1) {
            file_is_pcd = true;
        }
    } else if (filenames.size() == 1) {
        file_is_ply = true;
    } else {
        printUsage(argv[0]);
        return -1;
    }

    if (file_is_pcd) {
        if (pcl::io::loadPCDFile(argv[filenames[0]], *cloud) < 0) {
            std::cout << "Error loading point cloud " << argv[filenames[0]] << "\n";
            return -1;
        }
        pcl::console::print_info("\nFound pcd file.\n");
        pcl::console::print_info("[done, ");
        pcl::console::print_value("%g", tt.toc());
        pcl::console::print_info(" ms : ");
        pcl::console::print_value("%d", cloud->size());
        pcl::console::print_info(" points]\n");
    } else if (file_is_ply) {
        pcl::io::loadPLYFile(argv[filenames[0]], *cloud);
        if (cloud->points.size() <= 0 or cloud->points[0].x <= 0 and cloud->points[0].y <= 0 and
            cloud->points[0].z <= 0) {
            pcl::console::print_warn("\nloadPLYFile could not read the cloud, attempting to loadPolygonFile...\n");
            pcl::io::loadPolygonFile(argv[filenames[0]], cl);
            pcl::fromPCLPointCloud2(cl.cloud, *cloud);
            if (cloud->points.size() <= 0 or cloud->points[0].x <= 0 and cloud->points[0].y <= 0 and
                cloud->points[0].z <= 0) {
                pcl::console::print_warn("\nloadPolygonFile could not read the cloud, attempting to PLYReader...\n");
                pcl::PLYReader plyRead;
                plyRead.read(argv[filenames[0]], *cloud);
                if (cloud->points.size() <= 0 or cloud->points[0].x <= 0 and cloud->points[0].y <= 0 and
                    cloud->points[0].z <= 0) {
                    pcl::console::print_error("\nError. ply file is not compatible.\n");
                    return -1;
                }
            }
        }

        pcl::console::print_info("\nFound ply file.");
        pcl::console::print_info("[done, ");
        pcl::console::print_value("%g", tt.toc());
        pcl::console::print_info(" ms : ");
        pcl::console::print_value("%d", cloud->size());
        pcl::console::print_info(" points]\n");

    } else if (file_is_txt) {
        std::ifstream file(argv[filenames[0]]);
        if (!file.is_open()) {
            std::cout << "Error: Could not find " << argv[filenames[0]] << std::endl;
            return -1;
        }

        std::cout << "file opened." << std::endl;
        double x_, y_, z_;
        unsigned int r, g, b;

        while (file >> x_ >> y_ >> z_ >> r >> g >> b) {
            pcl::PointXYZRGB pt;
            pt.x = x_;
            pt.y = y_;
            pt.z = z_;

            uint8_t r_, g_, b_;
            r_ = uint8_t(r);
            g_ = uint8_t(g);
            b_ = uint8_t(b);

            uint32_t rgb_ = ((uint32_t) r_ << 16 | (uint32_t) g_ << 8 | (uint32_t) b_);
            pt.rgb = *reinterpret_cast<float *>(&rgb_);

            cloud->points.push_back(pt);
            //std::cout << "pointXYZRGB:" <<  pt << std::endl;
        }

        pcl::console::print_info("\nFound txt file.\n");
        pcl::console::print_info("[done, ");
        pcl::console::print_value("%g", tt.toc());
        pcl::console::print_info(" ms : ");
        pcl::console::print_value("%d", cloud->points.size());
        pcl::console::print_info(" points]\n");

    } else if (file_is_xyz) {

        std::ifstream file(argv[filenames[0]]);
        if (!file.is_open()) {
            std::cout << "Error: Could not find " << argv[filenames[0]] << std::endl;
            return -1;
        }

        std::cout << "file opened." << std::endl;
        double x_, y_, z_;

        while (file >> x_ >> y_ >> z_) {

            pcl::PointXYZRGB pt;
            pt.x = x_;
            pt.y = y_;
            pt.z = z_;

            cloud->points.push_back(pt);
            //std::cout << "pointXYZRGB:" <<  pt << std::endl;
        }

        pcl::console::print_info("\nFound xyz file.\n");
        pcl::console::print_info("[done, ");
        pcl::console::print_value("%g", tt.toc());
        pcl::console::print_info(" ms : ");
        pcl::console::print_value("%d", cloud->points.size());
        pcl::console::print_info(" points]\n");
    }

    cloud->width = (int) cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    std::string output_dir = argv[2];
    std::string select_setKSearch = argv[3];
    std::string select_nThreads = argv[4];
    std::string select_depth = argv[5];
    std::string select_pointWeight = argv[6];
    std::string select_samplePNode = argv[7];
    std::string select_scale = argv[8];
    std::string select_isoDivide = argv[9];
    std::string select_degree = argv[10];

    int setKSearch = std::atoi(select_setKSearch.c_str());
    int nThreads = std::atoi(select_nThreads.c_str());
    int depth = std::atoi(select_depth.c_str());
    float pointWeight = std::atof(select_pointWeight.c_str());
    float samplePNode = std::atof(select_samplePNode.c_str());
    float scale = std::atof(select_scale.c_str());
    int isoDivide = std::atoi(select_isoDivide.c_str());
    int degree = std::atoi(select_degree.c_str());

//    std::string select_mode = argv[2];
//    std::string select_normal_method = argv[3];//10
//    std::string output_dir = argv[4];//10
//
//    int surface_mode = std::atoi(select_mode.c_str());
//    int normal_method = std::atoi(select_normal_method.c_str());



    boost::filesystem::path dirPath(output_dir);

    if (not boost::filesystem::exists(dirPath) or not boost::filesystem::is_directory(dirPath)) {
        pcl::console::print_error("\nError. does not exist or it's not valid: ");
        std::cout << output_dir << std::endl;
        std::exit(-1);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::copyPointCloud(*cloud, *cloud_xyz);

    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_filtered (new pcl::PointCloud<pcl::PointXYZ>());
    //cloudPointFilter(cloud_xyz,cloud_xyz_filtered);

    pcl::PolygonMesh cloud_mesh;
    create_mesh(cloud_xyz, cloud_mesh);

    output_dir += "/" + filenames + ".ply";

    std::string sav = "saved mesh in:";
    sav += output_dir;

    //typedef pcl::geometry::DefaultMeshTraits <>      MeshTraits;
    //typedef pcl::geometry::TriangleMesh <MeshTraits> Mesh;
    //typedef pcl::geometry::MeshIO <Mesh>             MeshIO;

    pcl::console::print_info(sav.c_str());
    std::cout << std::endl;

    pcl::io::savePLYFileBinary(output_dir.c_str(), cloud_mesh);
    //pcl::io::savePolygonFilePLY(output_dir.c_str(),cloud_mesh,true);


    vtkObject::GlobalWarningDisplayOff(); // Disable vtk render warning
//  vizualizeMesh(cloud,cloud_mesh);

    return 0;
}

