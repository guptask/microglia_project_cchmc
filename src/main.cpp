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


#define DEBUG_FLAG              0   // Debug flag for image channels
#define MICROGLIAL_ROI_FACTOR   20  // ROI of microglial cell = roi factor * mean microglial dia
#define NUM_AREA_BINS           21  // Number of bins
#define BIN_AREA                25  // Bin area
#define NUM_Z_LAYERS_COMBINED   1   // Number of z-layers combined


/* Channel type */
enum class ChannelType : unsigned char {
    BLUE = 0,
    GREEN,
    RED,
    RED_LOW,
    RED_HIGH
};

/* Hierarchy type */
enum class HierarchyType : unsigned char {
    INVALID_CNTR = 0,
    CHILD_CNTR,
    PARENT_CNTR
};

/* Enhance the image */
bool enhanceImage(cv::Mat src, ChannelType channel_type, cv::Mat *dst) {

    // Enhance the image using Gaussian blur and thresholding
    cv::Mat enhanced;
    switch(channel_type) {
        case ChannelType::BLUE: {
            // Enhance the blue channel

            // Create the mask
            cv::Mat src_gray;
            cv::threshold(src, src_gray, 10, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 150, 255, cv::THRESH_BINARY);

            // Invert the mask
            bitwise_not(enhanced, enhanced);
        } break;

        case ChannelType::GREEN: {
            // Enhance the green channel

            // Create the mask
            cv::Mat src_gray;
            cv::threshold(src, src_gray, 10, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 240, 255, cv::THRESH_BINARY);

            // Invert the mask
            bitwise_not(enhanced, enhanced);
        } break;

        case ChannelType::RED: {
            // Enhance the red channel

            // Create the mask
            cv::Mat src_gray;
            cv::threshold(src, src_gray, 5, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 250, 255, cv::THRESH_BINARY);

            // Invert the mask
            bitwise_not(enhanced, enhanced);
        } break;

        case ChannelType::RED_LOW: {
            // Enhance the red (low) channel

            // Create the mask
            cv::Mat src_gray;
            cv::threshold(src, src_gray, 50, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 250, 255, cv::THRESH_BINARY);

            // Enhance the low intensity features
            cv::Mat red_low_gauss;
            cv::GaussianBlur(src, red_low_gauss, cv::Size(3,3), 0, 0);
            bitwise_and(red_low_gauss, enhanced, enhanced);
            cv::threshold(enhanced, enhanced, 250, 255, cv::THRESH_TOZERO_INV);
            cv::threshold(enhanced, enhanced, 1, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::RED_HIGH: {
            // Enhance the red (high) channel

            // Create the mask
            cv::Mat src_gray;
            cv::threshold(src, src_gray, 50, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 250, 255, cv::THRESH_BINARY);

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
        case ChannelType::BLUE :
        case ChannelType::GREEN : {
            findContours(temp_src, *contours, *hierarchy, cv::RETR_EXTERNAL, 
                                                        cv::CHAIN_APPROX_SIMPLE);
        } break;

        case ChannelType::RED :
        case ChannelType::RED_LOW :
        case ChannelType::RED_HIGH : {
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

/* Classify Microglial cells */
void classifyMicroglialCells(std::vector<std::vector<cv::Point>> blue_contours, 
                                cv::Mat blue_red_intersection,
                                std::vector<std::vector<cv::Point>> *microglial_contours,
                                std::vector<std::vector<cv::Point>> *other_contours) {

    for (size_t i = 0; i < blue_contours.size(); i++) {

        // Eliminate small contours via contour arc calculation
        if ((arcLength(blue_contours[i], true) < 10) || (blue_contours[i].size() < 5)) continue;

        // Determine whether microglial cell by calculating blue-red coverage area
        std::vector<std::vector<cv::Point>> specific_contour (1, blue_contours[i]);
        cv::Mat drawing = cv::Mat::zeros(blue_red_intersection.size(), CV_8UC1);
        drawContours(drawing, specific_contour, -1, cv::Scalar::all(255), cv::FILLED, 
                                    cv::LINE_8, std::vector<cv::Vec4i>(), 0, cv::Point());
        int contour_count_before = countNonZero(drawing);
        cv::Mat contour_intersection;
        bitwise_and(drawing, blue_red_intersection, contour_intersection);
        int contour_count_after = countNonZero(contour_intersection);
        float coverage_ratio = ((float)contour_count_after)/contour_count_before;
        if (coverage_ratio < 0.30) {
            other_contours->push_back(blue_contours[i]);
        } else {
            microglial_contours->push_back(blue_contours[i]);
        }
    }
}

/* Classify Neural cells */
void classifyNeuralCells(std::vector<std::vector<cv::Point>> blue_contours, 
                            cv::Mat blue_green_intersection,
                            std::vector<std::vector<cv::Point>> *neural_contours,
                            std::vector<std::vector<cv::Point>> *other_contours) {

    for (size_t i = 0; i < blue_contours.size(); i++) {

        // Eliminate small contours via contour arc calculation
        if ((arcLength(blue_contours[i], true) < 10) || (blue_contours[i].size() < 5)) continue;

        // Determine whether neural cell by calculating blue-green coverage area
        std::vector<std::vector<cv::Point>> specific_contour (1, blue_contours[i]);
        cv::Mat drawing = cv::Mat::zeros(blue_green_intersection.size(), CV_8UC1);
        drawContours(drawing, specific_contour, -1, cv::Scalar::all(255), cv::FILLED, 
                                    cv::LINE_8, std::vector<cv::Vec4i>(), 0, cv::Point());
        int contour_count_before = countNonZero(drawing);
        cv::Mat contour_intersection;
        bitwise_and(drawing, blue_green_intersection, contour_intersection);
        int contour_count_after = countNonZero(contour_intersection);
        float coverage_ratio = ((float)contour_count_after)/contour_count_before;
        if (coverage_ratio < 0.20) {
            other_contours->push_back(blue_contours[i]);
        } else {
            neural_contours->push_back(blue_contours[i]);
        }
    }
}

/* Group microglia area into bins */
void binArea(std::vector<HierarchyType> contour_mask, 
                std::vector<double> contour_area, 
                std::string *contour_bins,
                unsigned int *contour_cnt) {

    std::vector<unsigned int> count(NUM_AREA_BINS, 0);
    *contour_cnt = 0;
    for (size_t i = 0; i < contour_mask.size(); i++) {
        if (contour_mask[i] != HierarchyType::PARENT_CNTR) continue;
        unsigned int area = static_cast<unsigned int>(round(contour_area[i]));
        unsigned int bin_index = 
            (area/BIN_AREA < NUM_AREA_BINS) ? area/BIN_AREA : NUM_AREA_BINS-1;
        count[bin_index]++;
    }

    for (size_t i = 0; i < count.size(); i++) {
        *contour_cnt += count[i];
        *contour_bins += std::to_string(count[i]) + ",";
    }
}

/* Process the images inside each directory */
bool processImage(std::string path, std::string image_name, std::string metrics_file) {

    /* Create the data output file for images that were processed */
    std::ofstream data_stream;
    data_stream.open(metrics_file, std::ios::app);
    if (!data_stream.is_open()) {
        std::cerr << "Could not open the data output file." << std::endl;
        return false;
    }

    // Count the number of images
    uint8_t z_count = 0;
    struct dirent *dir = NULL;

    std::string dir_name = path + "tiff/" + image_name + "/";
    DIR *read_dir = opendir(dir_name.c_str());
    if (!read_dir) {
        std::cerr << "Could not open directory '" << dir_name << "'" << std::endl;
        return false;
    }
    while ((dir = readdir(read_dir))) {
        if (!strcmp (dir->d_name, ".") || !strcmp (dir->d_name, "..")) {
            continue;
        }
        z_count++;
    }
    closedir(read_dir);

    // Create the output directory
    std::string out_directory = path + "result/";
    struct stat st = {0};
    if (stat(out_directory.c_str(), &st) == -1) {
        mkdir(out_directory.c_str(), 0700);
    }
    out_directory = out_directory + image_name + "/";
    st = {0};
    if (stat(out_directory.c_str(), &st) == -1) {
        mkdir(out_directory.c_str(), 0700);
    }

    std::vector<cv::Mat> blue(z_count), green(z_count), red(z_count), original(z_count);
    for (uint8_t z_index = 1; z_index <= z_count; z_index++) {

        // Create the input filename and rgb stream output filenames
        std::string in_filename;
        if (z_count < 10) {
            in_filename  = dir_name + image_name + "_z" + std::to_string(z_index) + "c1+2+3.tif";
        } else {
            if (z_index < 10) {
                in_filename  = dir_name + image_name + "_z0" + std::to_string(z_index) + "c1+2+3.tif";
            } else if (z_index < 100) {
                in_filename  = dir_name + image_name + "_z" + std::to_string(z_index) + "c1+2+3.tif";
            } else { // assuming number of z plane layers will never exceed 99
                std::cerr << "Does not support more than 99 z layers curently" << std::endl;
                return false;
            }
        }

        // Extract the bgr streams for each input image
        cv::Mat img = cv::imread(in_filename.c_str(), cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
        if (img.empty()) {
            std::cerr << "Invalid input filename" << std::endl;
            return false;
        }
        original[z_index-1] = img;

        std::vector<cv::Mat> channel(3);
        cv::split(img, channel);

        blue[z_index-1]  = channel[0];
        green[z_index-1] = channel[1];
        red[z_index-1]   = channel[2];

        // Original image
        std::string out_original = out_directory + 
            "layer_" + std::to_string(z_index) + "_a_original.tif";
        cv::imwrite(out_original.c_str(), img);
    }


    /** Gather BGR channel information needed for feature extraction **/

    cv::Mat blue_merge, green_merge, red_merge, red_low_merge, red_high_merge;
    uint8_t merged_layer_count = 0;

    for (uint8_t z_index = 0; z_index < z_count; z_index++) {
        cv::Mat blue_enhanced, green_enhanced, red_enhanced, 
                            red_low_enhanced, red_high_enhanced;

        if(!enhanceImage(blue[z_index], ChannelType::BLUE, &blue_enhanced)) {
            return false;
        }
        if(!enhanceImage(green[z_index], ChannelType::GREEN, &green_enhanced)) {
            return false;
        }
        if(!enhanceImage(red[z_index], ChannelType::RED, &red_enhanced)) {
            return false;
        }
        if(!enhanceImage(red[z_index], ChannelType::RED_LOW, &red_low_enhanced)) {
            return false;
        }
        if(!enhanceImage(red[z_index], ChannelType::RED_HIGH, &red_high_enhanced)) {
            return false;
        }
        if (z_index%NUM_Z_LAYERS_COMBINED) {
            bitwise_or(blue_enhanced, blue_merge, blue_merge);
            bitwise_or(green_enhanced, green_merge, green_merge);
            bitwise_or(red_enhanced, red_merge, red_merge);
            bitwise_or(red_low_enhanced, red_low_merge, red_low_merge);
            bitwise_or(red_high_enhanced, red_high_merge, red_high_merge);
        } else {
            blue_merge = blue_enhanced;
            green_merge = green_enhanced;
            red_merge = red_enhanced;
            red_low_merge = red_low_enhanced;
            red_high_merge = red_high_enhanced;
        }

        if (((z_index+1)%NUM_Z_LAYERS_COMBINED == 0) || (z_index+1 == z_count)) {

            merged_layer_count++;

            // Blue channel
            std::string out_blue = out_directory + 
                "blue_merged_layer_" + std::to_string(merged_layer_count) + "_enhanced.tif";
            if (DEBUG_FLAG) cv::imwrite(out_blue.c_str(), blue_merge);

            cv::Mat blue_segmented;
            std::vector<std::vector<cv::Point>> contours_blue;
            std::vector<cv::Vec4i> hierarchy_blue;
            std::vector<HierarchyType> blue_contour_mask;
            std::vector<double> blue_contour_area;
            contourCalc(blue_merge, ChannelType::BLUE, 1.0, &blue_segmented, 
                &contours_blue, &hierarchy_blue, &blue_contour_mask, &blue_contour_area);
            out_blue.insert(out_blue.find_last_of("."), "_segmented", 10);
            if (DEBUG_FLAG) cv::imwrite(out_blue.c_str(), blue_segmented);

            // Green channel
            std::string out_green = out_directory + 
                "green_merged_layer_" + std::to_string(merged_layer_count) + "_enhanced.tif";
            if (DEBUG_FLAG) cv::imwrite(out_green.c_str(), green_merge);

            cv::Mat green_segmented;
            std::vector<std::vector<cv::Point>> contours_green;
            std::vector<cv::Vec4i> hierarchy_green;
            std::vector<HierarchyType> green_contour_mask;
            std::vector<double> green_contour_area;
            contourCalc(green_merge, ChannelType::GREEN, 1.0, &green_segmented, 
                &contours_green, &hierarchy_green, &green_contour_mask, &green_contour_area);
            out_green.insert(out_green.find_last_of("."), "_segmented", 10);
            if (DEBUG_FLAG) cv::imwrite(out_green.c_str(), green_segmented);

            // Red channel
            std::string out_red = out_directory + 
                "red_merged_layer_" + std::to_string(merged_layer_count) + "_enhanced.tif";
            if (DEBUG_FLAG) cv::imwrite(out_red.c_str(), red_merge);

            cv::Mat red_segmented;
            std::vector<std::vector<cv::Point>> contours_red;
            std::vector<cv::Vec4i> hierarchy_red;
            std::vector<HierarchyType> red_contour_mask;
            std::vector<double> red_contour_area;
            contourCalc(red_merge, ChannelType::RED, 1.0, &red_segmented, 
                &contours_red, &hierarchy_red, &red_contour_mask, &red_contour_area);
            out_red.insert(out_red.find_last_of("."), "_segmented", 10);
            if (DEBUG_FLAG) cv::imwrite(out_red.c_str(), red_segmented);

            // Red (low) channel
            std::string out_red_low = out_directory + 
                "red_low_merged_layer_" + std::to_string(merged_layer_count) + "_enhanced.tif";
            if (DEBUG_FLAG) cv::imwrite(out_red_low.c_str(), red_low_merge);

            cv::Mat red_low_segmented;
            std::vector<std::vector<cv::Point>> contours_red_low;
            std::vector<cv::Vec4i> hierarchy_red_low;
            std::vector<HierarchyType> red_low_contour_mask;
            std::vector<double> red_low_contour_area;
            contourCalc(red_low_merge, ChannelType::RED_LOW, 1.0, &red_low_segmented, 
                &contours_red_low, &hierarchy_red_low, &red_low_contour_mask, &red_low_contour_area);
            out_red.insert(out_red_low.find_last_of("."), "_segmented", 10);
            if (DEBUG_FLAG) cv::imwrite(out_red_low.c_str(), red_low_segmented);

            // Red (high) channel
            std::string out_red_high = out_directory + 
                "red_high_merged_layer_" + std::to_string(merged_layer_count) + "_enhanced.tif";
            if (DEBUG_FLAG) cv::imwrite(out_red_high.c_str(), red_high_merge);

            cv::Mat red_high_segmented;
            std::vector<std::vector<cv::Point>> contours_red_high;
            std::vector<cv::Vec4i> hierarchy_red_high;
            std::vector<HierarchyType> red_high_contour_mask;
            std::vector<double> red_high_contour_area;
            contourCalc(red_high_merge, ChannelType::RED_HIGH, 1.0, &red_high_segmented, 
                &contours_red_high, &hierarchy_red_high, &red_high_contour_mask, &red_high_contour_area);
            out_red.insert(out_red_high.find_last_of("."), "_segmented", 10);
            if (DEBUG_FLAG) cv::imwrite(out_red_high.c_str(), red_high_segmented);


            /** Extract multi-dimensional features for analysis **/

            // Blue-red channel intersection
            cv::Mat blue_red_intersection;
            bitwise_and(blue_merge, red_merge, blue_red_intersection);
            std::string out_blue_red_intersection = out_directory + 
                "blue_red_merged_layer_" + std::to_string(merged_layer_count) + "_enhanced.tif";
            if (DEBUG_FLAG) cv::imwrite(out_blue_red_intersection.c_str(), 
                                                        blue_red_intersection);

            // Classify microglial cells
            std::vector<std::vector<cv::Point>> microglial_contours, other_contours;
            classifyMicroglialCells(contours_blue, blue_red_intersection, 
                                        &microglial_contours, &other_contours);
            data_stream << image_name + "_" + std::to_string(merged_layer_count) << "," 
                        << microglial_contours.size() + other_contours.size() << "," 
                        << microglial_contours.size() << ",";

            // Blue-green channel intersection
            cv::Mat blue_green_intersection;
            bitwise_and(blue_merge, green_merge, blue_green_intersection);
            std::string out_blue_green_intersection = out_directory + 
                "blue_green_merged_layer_" + std::to_string(merged_layer_count) + "_enhanced.tif";
            if (DEBUG_FLAG) cv::imwrite(out_blue_green_intersection.c_str(), 
                                                        blue_green_intersection);

            // Classify neural cells
            std::vector<std::vector<cv::Point>> neural_contours, remaining_contours;
            classifyNeuralCells(other_contours, blue_green_intersection, 
                                    &neural_contours, &remaining_contours);
            data_stream << neural_contours.size() << "," 
                        << remaining_contours.size() << ",";

            // Characterize microglial cells
            std::string microglial_bins;
            unsigned int microglial_cnt;
            binArea(red_contour_mask, red_contour_area, &microglial_bins, &microglial_cnt);
            data_stream << microglial_cnt << "," << microglial_bins;

            // Green-red channel intersection
            cv::Mat green_red_intersection;
            bitwise_and(green_merge, red_merge, green_red_intersection);
            std::string out_green_red_intersection = out_directory + 
                "green_red_merged_layer_" + std::to_string(merged_layer_count) + "_enhanced.tif";
            if (DEBUG_FLAG) cv::imwrite(out_green_red_intersection.c_str(), green_red_intersection);

            // Segment the green-red intersection
            cv::Mat green_red_segmented;
            std::vector<std::vector<cv::Point>> contours_green_red;
            std::vector<cv::Vec4i> hierarchy_green_red;
            std::vector<HierarchyType> green_red_contour_mask;
            std::vector<double> green_red_contour_area;
            contourCalc(green_red_intersection, ChannelType::RED, 1.0, &green_red_segmented, 
                        &contours_green_red, &hierarchy_green_red, &green_red_contour_mask, 
                        &green_red_contour_area);

            // Characterize microglial fibre interaction with neural cells
            std::string microglial_neural_bins;
            unsigned int microglial_neural_cnt;
            binArea(green_red_contour_mask, green_red_contour_area, 
                    &microglial_neural_bins, &microglial_neural_cnt);
            data_stream << microglial_neural_cnt << "," << microglial_neural_bins;

            // Characterize high intensity microglial fibres
            std::string red_high_bins;
            unsigned int red_high_cnt;
            binArea(red_high_contour_mask, red_high_contour_area, &red_high_bins, &red_high_cnt);
            data_stream << red_high_cnt << "," << red_high_bins;

            // Characterize low intensity microglial fibres
            std::string red_low_bins;
            unsigned int red_low_cnt;
            binArea(red_low_contour_mask, red_low_contour_area, &red_low_bins, &red_low_cnt);
            data_stream << red_low_cnt << "," << red_low_bins;


            data_stream << std::endl;


            /** Enhanced image **/

            std::vector<cv::Mat> merge_enhanced;
            merge_enhanced.push_back(blue_merge);
            merge_enhanced.push_back(green_merge);
            merge_enhanced.push_back(red_merge);
            cv::Mat color_enhanced;
            cv::merge(merge_enhanced, color_enhanced);
            std::string out_enhanced = out_directory + 
                "layer_" + std::to_string(merged_layer_count) + "_b_enhanced.tif";
            cv::imwrite(out_enhanced.c_str(), color_enhanced);


            /** Analyzed image **/

            cv::Mat drawing_blue  = blue_merge;
            cv::Mat drawing_green = cv::Mat::zeros(green_merge.size(), CV_8UC1);
            cv::Mat drawing_red   = cv::Mat::zeros(red_merge.size(), CV_8UC1);

            // Draw microglial cell boundaries
            for (size_t i = 0; i < microglial_contours.size(); i++) {
                cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(microglial_contours[i]));
                ellipse(drawing_blue, min_ellipse, 255, 4, 8);
                ellipse(drawing_green, min_ellipse, 0, 4, 8);
                ellipse(drawing_red, min_ellipse, 255, 4, 8);
            }

            // Draw neural cell boundaries
            for (size_t i = 0; i < neural_contours.size(); i++) {
                cv::RotatedRect min_ellipse = fitEllipse(cv::Mat(neural_contours[i]));
                ellipse(drawing_blue, min_ellipse, 255, 4, 8);
                ellipse(drawing_green, min_ellipse, 255, 4, 8);
                ellipse(drawing_red, min_ellipse, 0, 4, 8);
            }

            // Merge the modified red, blue and green layers
            std::vector<cv::Mat> merge_analyzed;
            merge_analyzed.push_back(drawing_blue);
            merge_analyzed.push_back(drawing_green);
            merge_analyzed.push_back(drawing_red);
            cv::Mat color_analyzed;
            cv::merge(merge_analyzed, color_analyzed);
            std::string out_analyzed = out_directory + 
                "layer_" + std::to_string(merged_layer_count) + "_c_analyzed.tif";
            cv::imwrite(out_analyzed.c_str(), color_analyzed);
        }
    }
    data_stream.close();
    return true;
}

/* Main - create the threads and start the processing */
int main(int argc, char *argv[]) {

    /* Check for argument count */
    if (argc != 2) {
        std::cerr << "Invalid number of arguments." << std::endl;
        return -1;
    }

    /* Read the path to the data */
    std::string path(argv[1]);

    /* Read the list of directories to process */
    std::string image_list_filename = path + "image_list.dat";
    std::vector<std::string> input_images;
    FILE *file = fopen(image_list_filename.c_str(), "r");
    if (!file) {
        std::cerr << "Could not open 'image_list.dat' inside '" << path << "'." << std::endl;
        return -1;
    }
    char line[128];
    while (fgets(line, sizeof(line), file) != NULL) {
        line[strlen(line)-1] = 0;
        std::string temp_str(line);
        input_images.push_back(temp_str);
    }
    fclose(file);

    /* Create the error log for images that could not be processed */
    std::string err_file = path + "err_list.dat";
    std::ofstream err_stream(err_file);
    if (!err_stream.is_open()) {
        std::cerr << "Could not open the error log file." << std::endl;
        return -1;
    }

    /* Create and prepare the file for metrics */
    std::string metrics_file = path + "computed_metrics.csv";
    std::ofstream data_stream;
    data_stream.open(metrics_file, std::ios::out);
    if (!data_stream.is_open()) {
        std::cerr << "Could not create the metrics file." << std::endl;
        return -1;
    }

    data_stream << "image_layer,total nuclei count,microglial nuclei count,\
                neural nuclei count,other nuclei count,microglial fibre count,";

    for (unsigned int i = 0; i < NUM_AREA_BINS-1; i++) {
        data_stream << i*BIN_AREA << " <= microglial fibre area < " 
                    << (i+1)*BIN_AREA << ",";
    }
    data_stream << "microglial fibre area >= " 
                << (NUM_AREA_BINS-1)*BIN_AREA << ",";

    data_stream << "microglial fibre - neural cell intersection count,";
    for (unsigned int i = 0; i < NUM_AREA_BINS-1; i++) {
        data_stream << i*BIN_AREA 
                    << " <= microglial fibre - neural cell intersection area < " 
                    << (i+1)*BIN_AREA << ",";
    }
    data_stream << "microglial fibre - neural cell intersection area >= " 
                << (NUM_AREA_BINS-1)*BIN_AREA << ",";

    data_stream << "high intensity microglial fibre count,";
    for (unsigned int i = 0; i < NUM_AREA_BINS-1; i++) {
        data_stream << i*BIN_AREA 
                    << " <= high intensity microglial fibre area < " 
                    << (i+1)*BIN_AREA << ",";
    }
    data_stream << "high intensity microglial fibre area >= " 
                << (NUM_AREA_BINS-1)*BIN_AREA << ",";

    data_stream << "low intensity microglial fibre count,";
    for (unsigned int i = 0; i < NUM_AREA_BINS-1; i++) {
        data_stream << i*BIN_AREA 
                    << " <= low intensity microglial fibre area < " 
                    << (i+1)*BIN_AREA << ",";
    }
    data_stream << "low intensity microglial fibre area >= " 
                << (NUM_AREA_BINS-1)*BIN_AREA << ",";

    data_stream << std::endl;
    data_stream.close();

    /* Process each image directory */
    for (unsigned int index = 0; index < input_images.size(); index++) {
        std::cout << "Processing " << input_images[index] << std::endl;
        if (!processImage(path, input_images[index], metrics_file)) {
            err_stream << input_images[index] << std::endl;
        }
    }
    err_stream.close();

    return 0;
}

