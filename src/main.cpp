#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <math.h>
#include <assert.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/imgcodecs.hpp"

#define NUM_Z_LAYERS            3   // Merge a certain number of z layers
#define DEBUG_FLAG              1   // Debug flag for image channels

/* Channel type */
enum class ChannelType : unsigned char {
    BLUE = 0,
    GREEN,
    RED
};

/* Hierarchy type */
enum class HierarchyType : unsigned char {
    INVALID_CNTR = 0,
    CHILD_CNTR,
    PARENT_CNTR
};

/* Enhance the image */
bool enhanceImage(cv::Mat src, ChannelType channel_type, cv::Mat *dst) {

    // Convert to grayscale
    cv::Mat src_gray;
    cvtColor (src, src_gray, cv::COLOR_BGR2GRAY);

    // Enhance the image using Gaussian blur and thresholding
    cv::Mat enhanced;
    switch(channel_type) {
        case ChannelType::BLUE: {
            // Enhance the blue channel

            // Create the mask
            cv::threshold(src_gray, src_gray, 50, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 240, 255, cv::THRESH_BINARY);

            // Invert the mask
            bitwise_not(enhanced, enhanced);
        } break;

        case ChannelType::GREEN: {
            // Enhance the green channel

            // Create the mask
            cv::threshold(src_gray, src_gray, 50, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 240, 255, cv::THRESH_BINARY);

            // Invert the mask
            bitwise_not(enhanced, enhanced);
        } break;

        case ChannelType::RED: {
            // Enhance the red channel

            // Create the mask
            cv::threshold(src_gray, src_gray, 5, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 240, 255, cv::THRESH_BINARY);

            // Invert the mask
            bitwise_not(enhanced, enhanced);
        } break;

        default: {
            std::cerr << "Invalid channel type" << std::endl;
            return false;
        }
    }
    *dst = enhanced;
    return true;
}

/* Find the contours in the image */
void contourCalc(cv::Mat src, ChannelType channel_type, 
                    double min_area, cv::Mat *dst, 
                    std::vector<std::vector<cv::Point>> *contours, 
                    std::vector<cv::Vec4i> *hierarchy, 
                    std::vector<HierarchyType> *validity_mask, 
                    std::vector<double> *parent_area) {

    cv::Mat temp_src;
    src.copyTo(temp_src);
    switch(channel_type) {
        case ChannelType::BLUE:
        case ChannelType::GREEN: {
            findContours(temp_src, *contours, *hierarchy, cv::RETR_EXTERNAL, 
                                                        cv::CHAIN_APPROX_SIMPLE);
        } break;

        case ChannelType::RED : {
            findContours(temp_src, *contours, *hierarchy, cv::RETR_CCOMP, 
                                                        cv::CHAIN_APPROX_SIMPLE);
        } break;

        default: return;
    }

    *dst = cv::Mat::zeros(temp_src.size(), CV_8UC3);
    if (!contours->size()) return;
    validity_mask->assign(contours->size(), HierarchyType::INVALID_CNTR);
    parent_area->assign(contours->size(), 0.0);

    // Keep the contours whose size is >= than min_area
    cv::RNG rng(12345);
    for (int index = 0 ; index < (int)contours->size(); index++) {
        if ((*hierarchy)[index][3] > -1) continue; // ignore child
        auto cntr_external = (*contours)[index];
        double area_external = fabs(contourArea(cv::Mat(cntr_external)));
        if (area_external < min_area) continue;

        std::vector<int> cntr_list;
        cntr_list.push_back(index);

        int index_hole = (*hierarchy)[index][2];
        double area_hole = 0.0;
        while (index_hole > -1) {
            std::vector<cv::Point> cntr_hole = (*contours)[index_hole];
            double temp_area_hole = fabs(contourArea(cv::Mat(cntr_hole)));
            if (temp_area_hole) {
                cntr_list.push_back(index_hole);
                area_hole += temp_area_hole;
            }
            index_hole = (*hierarchy)[index_hole][0];
        }
        double area_contour = area_external - area_hole;
        if (area_contour >= min_area) {
            (*validity_mask)[cntr_list[0]] = HierarchyType::PARENT_CNTR;
            (*parent_area)[cntr_list[0]] = area_contour;
            for (unsigned int i = 1; i < cntr_list.size(); i++) {
                (*validity_mask)[cntr_list[i]] = HierarchyType::CHILD_CNTR;
            }
            cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), 
                                            rng.uniform(0,255));
            drawContours(*dst, *contours, index, color, cv::FILLED, cv::LINE_8, *hierarchy);
        }
    }
}

/* Process the images inside each directory */
bool processDir(std::string dir_name, std::string out_file) {

    /* Create the data output file for images that were processed */
    std::ofstream data_stream;
    data_stream.open(out_file, std::ios::app);
    if (!data_stream.is_open()) {
        std::cerr << "Could not open the data output file." << std::endl;
        return false;
    }

    // Create a alternative directory name for the data collection
    // Replace '/' and ' ' with '_'
    std::string dir_name_modified = dir_name;
    std::size_t found = dir_name_modified.find("/");
    while (found != std::string::npos) {
        dir_name_modified.replace(found, 1, "_");
        found = dir_name_modified.find("/");
    }

    DIR *read_dir = opendir(dir_name.c_str());
    if (!read_dir) {
        std::cerr << "Could not open directory '" << dir_name << "'" << std::endl;
        return false;
    }

    // Count the number of images
    uint8_t z_count = 0;
    struct dirent *dir = NULL;
    while ((dir = readdir(read_dir))) {
        if (!strcmp (dir->d_name, ".") || !strcmp (dir->d_name, "..")) {
            continue;
        }
        z_count++;
    }
    closedir(read_dir);

    if ((z_count < NUM_Z_LAYERS) || (NUM_Z_LAYERS > 3)) {
        std::cerr << "Not enough z layers in the image." << std::endl;
        return false;
    }

    // Extract the input directory name
    std::istringstream iss(dir_name);
    std::string token;
    getline(iss, token, '/');
    getline(iss, token, '/');

    // Create the output directory
    std::string out_directory = "result/" + token + "/";
    struct stat st = {0};
    if (stat(out_directory.c_str(), &st) == -1) {
        mkdir(out_directory.c_str(), 0700);
    }

    std::vector<cv::Mat> blue(NUM_Z_LAYERS), green(NUM_Z_LAYERS), 
                                red(NUM_Z_LAYERS), original(NUM_Z_LAYERS);
    for (uint8_t z_index = 1; z_index <= z_count; z_index++) {

        // Create the input filename and rgb stream output filenames
        std::string in_filename;
        if (z_count < 10) {
            in_filename  = dir_name + token + "_z" + std::to_string(z_index) + "c1+2+3.tif";
        } else {
            if (z_index < 10) {
                in_filename  = dir_name + token + "_z0" + std::to_string(z_index) + "c1+2+3.tif";
            } else if (z_index < 100) {
                in_filename  = dir_name + token + "_z" + std::to_string(z_index) + "c1+2+3.tif";
            } else { // assuming number of z plane layers will never exceed 99
                std::cerr << "Does not support more than 99 z layers curently" << std::endl;
                return false;
            }
        }

        // Extract the bgr streams for each input image
        cv::Mat img = cv::imread(in_filename.c_str());
        if (img.empty()) {
            std::cerr << "Invalid input filename" << std::endl;
            return false;
        }
        original[(z_index-1)%NUM_Z_LAYERS] = img;

        std::vector<cv::Mat> channel(3);
        cv::split(img, channel);

        blue[(z_index-1)%NUM_Z_LAYERS] = channel[0];
        green[(z_index-1)%NUM_Z_LAYERS] = channel[1];
        red[(z_index-1)%NUM_Z_LAYERS] = channel[2];

        // Manipulate RGB channels and extract features for a certain number of Z layers
        if (z_index >= NUM_Z_LAYERS) {

            /* Gather BGR channel information needed for feature extraction */

            // Blue channel
            cv::Mat blue_merge, blue_enhanced, blue_segmented;
            std::vector<std::vector<cv::Point>> contours_blue;
            std::vector<cv::Vec4i> hierarchy_blue;
            std::vector<HierarchyType> blue_contour_mask;
            std::vector<double> blue_contour_area;

            cv::merge(blue, blue_merge);
            std::string out_blue = out_directory + "trilayer_blue_layer_" + 
                            std::to_string(z_index-NUM_Z_LAYERS+1) + ".tif";
            if (DEBUG_FLAG) cv::imwrite(out_blue.c_str(), blue_merge);
            if(!enhanceImage(blue_merge, ChannelType::BLUE, &blue_enhanced)) {
                return false;
            }
            out_blue.insert(out_blue.find_last_of("."), "_enhanced", 9);
            if (DEBUG_FLAG) cv::imwrite(out_blue.c_str(), blue_enhanced);
            contourCalc(blue_enhanced, ChannelType::BLUE, 1.0, &blue_segmented, 
                            &contours_blue, &hierarchy_blue, &blue_contour_mask, 
                            &blue_contour_area);
            out_blue.insert(out_blue.find_last_of("."), "_segmented", 10);
            if (DEBUG_FLAG) cv::imwrite(out_blue.c_str(), blue_segmented);

            // Green channel
            cv::Mat green_merge, green_enhanced, green_segmented;
            std::vector<std::vector<cv::Point>> contours_green;
            std::vector<cv::Vec4i> hierarchy_green;
            std::vector<HierarchyType> green_contour_mask;
            std::vector<double> green_contour_area;

            cv::merge(green, green_merge);
            std::string out_green = out_directory + "trilayer_green_layer_" + 
                            std::to_string(z_index-NUM_Z_LAYERS+1) + ".tif";
            if (DEBUG_FLAG) cv::imwrite(out_green.c_str(), green_merge);
            if(!enhanceImage(green_merge, ChannelType::GREEN, &green_enhanced)) {
                return false;
            }
            out_green.insert(out_green.find_last_of("."), "_enhanced", 9);
            if (DEBUG_FLAG) cv::imwrite(out_green.c_str(), green_enhanced);
            contourCalc(green_enhanced, ChannelType::GREEN, 1.0, &green_segmented, 
                            &contours_green, &hierarchy_green, &green_contour_mask, 
                            &green_contour_area);
            out_green.insert(out_green.find_last_of("."), "_segmented", 10);
            if (DEBUG_FLAG) cv::imwrite(out_green.c_str(), green_segmented);

            // Red channel
            cv::Mat red_merge, red_enhanced, red_segmented;
            std::vector<std::vector<cv::Point>> contours_red;
            std::vector<cv::Vec4i> hierarchy_red;
            std::vector<HierarchyType> red_contour_mask;
            std::vector<double> red_contour_area;

            cv::merge(red, red_merge);
            std::string out_red = out_directory + "trilayer_red_layer_" + 
                            std::to_string(z_index-NUM_Z_LAYERS+1) + ".tif";
            if (DEBUG_FLAG) cv::imwrite(out_red.c_str(), red_merge);
            if(!enhanceImage(red_merge, ChannelType::RED, &red_enhanced)) {
                return false;
            }
            out_red.insert(out_red.find_last_of("."), "_enhanced", 9);
            if (DEBUG_FLAG) cv::imwrite(out_red.c_str(), red_enhanced);
            contourCalc(red_enhanced, ChannelType::RED, 1.0, &red_segmented, 
                            &contours_red, &hierarchy_red, &red_contour_mask, 
                            &red_contour_area);
            out_red.insert(out_red.find_last_of("."), "_segmented", 10);
            if (DEBUG_FLAG) cv::imwrite(out_red.c_str(), red_segmented);


            /** Extract multi-dimensional features for analysis **/
            //TODO

        }
    }
    data_stream.close();

    return true;
}

/* Main - create the threads and start the processing */
int main(int argc, char *argv[]) {

    /* Check for argument count */
    if (argc != 5) {
        std::cerr << "Invalid number of arguments." << std::endl;
        return -1;
    }

    /* Read the path to the data */
    std::string path(argv[1]);

    /* Read the list of directories to process */
    std::vector<std::string> files;
    FILE *file = fopen(argv[2], "r");
    if (!file) {
        std::cerr << "Could not open the file list." << std::endl;
        return -1;
    }
    char line[128];
    while (fgets(line, sizeof(line), file) != NULL) {
        line[strlen(line)-1] = '/';
        std::string temp_str(line);
        std::string image_name = path + temp_str;
        files.push_back(image_name);
    }
    fclose(file);

    /* Create the error log for images that could not be processed */
    std::ofstream err_file(argv[3]);
    if (!err_file.is_open()) {
        std::cerr << "Could not open the error log file." << std::endl;
        return -1;
    }

    /* Create and prepare the file for metrics */
    std::string out_file(argv[4]);
    std::ofstream data_stream;
    data_stream.open(out_file, std::ios::out);
    if (!data_stream.is_open()) {
        std::cerr << "Could not create the data output file." << std::endl;
        return -1;
    }

    /* Process each image directory */
    for (auto& file_name : files) {
        std::cout << "Processing " << file_name << std::endl;
        if (!processDir(file_name, out_file)) {
            err_file << file_name << std::endl;
        }
    }
    err_file.close();

    return 0;
}

