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

//#include <pcl/surface/poisson.h>
//#include <pcl/surface/mls.h>
//#include <pcl/surface/simplification_remove_unused_vertices.h>
//#include <pcl/surface/vtk_smoothing/vtk_utils.h>
//#include <pcl/surface/gp3.h>
//#include <pcl/surface/convex_hull.h>

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
    std::cout << "\nUsage: " << progName << " <input cloud> <output dir> <setKSearch normal> <nThreads> <depth> <pointWeight f> <samplePNode f> <scale f> <isoDivide> <degree>"
              << std::endl;
}

void create_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_with_normals, int setKSearch) {

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
    // normal cal
    std::cout << "Using normal method estimation...";

    pcl::NormalEstimationOMP <pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);

    n.setInputCloud(cloudTranslated);
    n.setSearchMethod(kdtree_for_points);
    n.setKSearch(setKSearch); //It was 20
    n.compute(*normals);//Normals are estimated using standard method.

    //pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal> ());
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    std::cout << "normal estimated [OK]" << std::endl;

    // end normal estimation
}


int main(int argc, char **argv) {

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PolygonMesh cl;
    std::vector<int> filenames;
    bool file_is_pcd = false;
    bool file_is_ply = false;
    bool file_is_txt = false;
    bool file_is_xyz = false;

    if (argc < 4 or argc > 4) {
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

    int setKSearch = std::atoi(select_setKSearch.c_str());

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
    pcl::PointCloud<pcl::PointNormal>::Ptr xyz_normal(new pcl::PointCloud<pcl::PointNormal>());
    create_normal(cloud_xyz, xyz_normal, setKSearch);
    char* filename = argv[filenames[0]];
    output_dir += "/" + std::string(filename, filename+sizeof(filename)-3) + ".ply";

    std::string sav = "saved mesh in:";
    sav += output_dir;

    //typedef pcl::geometry::DefaultMeshTraits <>      MeshTraits;
    //typedef pcl::geometry::TriangleMesh <MeshTraits> Mesh;
    //typedef pcl::geometry::MeshIO <Mesh>             MeshIO;

    pcl::console::print_info(sav.c_str());
    std::cout << std::endl;

    pcl::io::savePLYFileBinary(output_dir.c_str(), *xyz_normal);
    //pcl::io::savePolygonFilePLY(output_dir.c_str(),cloud_mesh,true);


    vtkObject::GlobalWarningDisplayOff(); // Disable vtk render warning
//  vizualizeMesh(cloud,cloud_mesh);

    return 0;
}

