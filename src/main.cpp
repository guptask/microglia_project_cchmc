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


#define DEBUG_FLAG                0   // Debug flag for image channels
#define MICROGLIAL_ROI_FACTOR     20  // ROI of microglial cell = roi factor * mean microglial dia
#define NUM_MICROGLIA_AREA_BINS   21  // Number of bins
#define MICROGLIA_BIN_AREA        25  // Bin area


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

    // Enhance the image using Gaussian blur and thresholding
    cv::Mat enhanced;
    switch(channel_type) {
        case ChannelType::BLUE: {
            // Enhance the blue channel

            // Create the mask
            cv::Mat src_gray;
            cv::threshold(src, src_gray, 50, 255, cv::THRESH_TOZERO);
            bitwise_not(src_gray, src_gray);
            cv::GaussianBlur(src_gray, enhanced, cv::Size(3,3), 0, 0);
            cv::threshold(enhanced, enhanced, 240, 255, cv::THRESH_BINARY);

            // Invert the mask
            bitwise_not(enhanced, enhanced);
        } break;

        case ChannelType::GREEN: {
            // Enhance the green channel

            // Create the mask
            cv::Mat src_gray;
            cv::threshold(src, src_gray, 50, 255, cv::THRESH_TOZERO);
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
        if (coverage_ratio < 0.50) {
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
        if (coverage_ratio < 0.50) {
            other_contours->push_back(blue_contours[i]);
        } else {
            neural_contours->push_back(blue_contours[i]);
        }
    }
}

/* Group microglia area into bins */
void binMicrogliaArea(std::vector<HierarchyType> contour_mask, 
                        std::vector<double> contour_area, 
                        std::string *contour_bins,
                        unsigned int *contour_cnt) {

    std::vector<unsigned int> count(NUM_MICROGLIA_AREA_BINS, 0);
    *contour_cnt = 0;
    for (size_t i = 0; i < contour_mask.size(); i++) {
        if (contour_mask[i] != HierarchyType::PARENT_CNTR) continue;
        unsigned int area = static_cast<unsigned int>(round(contour_area[i]));
        unsigned int bin_index = (area/MICROGLIA_BIN_AREA < NUM_MICROGLIA_AREA_BINS) ? 
                                        area/MICROGLIA_BIN_AREA : NUM_MICROGLIA_AREA_BINS-1;
        count[bin_index]++;
    }

    for (size_t i = 0; i < count.size(); i++) {
        *contour_cnt += count[i];
        *contour_bins += std::to_string(count[i]) + ",";
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

    std::vector<cv::Mat> blue(z_count), green(z_count), red(z_count), original(z_count);
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
        original[z_index-1] = img;

        std::vector<cv::Mat> channel(3);
        cv::split(img, channel);

        blue[z_index-1]  = channel[0];
        green[z_index-1] = channel[1];
        red[z_index-1]   = channel[2];
    }


    /** Gather BGR channel information needed for feature extraction **/

    // Blue channel
    cv::Mat blue_merge;
    for (uint8_t z_index = 0; z_index < z_count; z_index++) {
        cv::Mat blue_enhanced;
        if(!enhanceImage(blue[z_index], ChannelType::BLUE, &blue_enhanced)) {
            return false;
        }
        if (z_index) {
            bitwise_or(blue_enhanced, blue_merge, blue_merge);
        } else {
            blue_merge = blue_enhanced;
        }
    }
    std::string out_blue = out_directory + "blue_layer_merged_enhanced.tif";
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
    cv::Mat green_merge;
    for (uint8_t z_index = 0; z_index < z_count; z_index++) {
        cv::Mat green_enhanced;
        if(!enhanceImage(green[z_index], ChannelType::GREEN, &green_enhanced)) {
            return false;
        }
        if (z_index) {
            bitwise_or(green_enhanced, green_merge, green_merge);
        } else {
            green_merge = green_enhanced;
        }
    }
    std::string out_green = out_directory + "green_layer_merged_enhanced.tif";
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
    cv::Mat red_merge;
    for (uint8_t z_index = 0; z_index < z_count; z_index++) {
        cv::Mat red_enhanced;
        if(!enhanceImage(red[z_index], ChannelType::RED, &red_enhanced)) {
            return false;
        }
        if (z_index) {
            bitwise_or(red_enhanced, red_merge, red_merge);
        } else {
            red_merge = red_enhanced;
        }
    }
    std::string out_red = out_directory + "red_layer_merged_enhanced.tif";
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


    /** Extract multi-dimensional features for analysis **/

    // Blue-red channel intersection
    cv::Mat blue_red_intersection;
    bitwise_and(blue_merge, red_merge, blue_red_intersection);
    std::string out_blue_red_intersection = out_directory + 
                        "blue_red_layers_merged_enhanced.tif";
    if (DEBUG_FLAG) cv::imwrite(out_blue_red_intersection.c_str(), blue_red_intersection);

    // Classify microglial cells
    std::vector<std::vector<cv::Point>> microglial_contours, other_contours;
    classifyMicroglialCells(contours_blue, blue_red_intersection, 
                            &microglial_contours, &other_contours);
    data_stream << dir_name << "," 
                << microglial_contours.size() + other_contours.size() << "," 
                << microglial_contours.size() << ",";

    // Blue-green channel intersection
    cv::Mat blue_green_intersection;
    bitwise_and(blue_merge, green_merge, blue_green_intersection);
    std::string out_blue_green_intersection = out_directory + 
                        "blue_green_layers_merged_enhanced.tif";
    if (DEBUG_FLAG) cv::imwrite(out_blue_green_intersection.c_str(), blue_green_intersection);

    // Classify neural cells
    std::vector<std::vector<cv::Point>> neural_contours, remaining_contours;
    classifyNeuralCells(other_contours, blue_green_intersection, 
                            &neural_contours, &remaining_contours);
    data_stream << neural_contours.size() << "," 
                << remaining_contours.size() << ",";

    // Characterize microglial cells
    std::string microglial_bins;
    unsigned int microglial_cnt;
    binMicrogliaArea(red_contour_mask, red_contour_area, &microglial_bins, &microglial_cnt);
    data_stream << microglial_cnt << "," << microglial_bins;


    data_stream << std::endl;
    data_stream.close();


    /** Original image **/

    std::vector<cv::Mat> merge_original;
    merge_original.push_back(blue_merge);
    merge_original.push_back(green_merge);
    merge_original.push_back(red_merge);
    cv::Mat color_original;
    cv::merge(merge_original, color_original);
    std::string out_original = out_directory + "original_enhanced_and_flatened.tif";
    cv::imwrite(out_original.c_str(), color_original);


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
    std::vector<cv::Mat> merge_analysis;
    merge_analysis.push_back(drawing_blue);
    merge_analysis.push_back(drawing_green);
    merge_analysis.push_back(drawing_red);
    cv::Mat color_analysis;
    cv::merge(merge_analysis, color_analysis);
    std::string out_analysis = out_directory + "cell_classification.tif";
    cv::imwrite(out_analysis.c_str(), color_analysis);

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

    data_stream << "image,total nuclei count,microglial nuclei count,\
                neural nuclei count,other nuclei count,microglia fibre count,";

    for (unsigned int i = 0; i < NUM_MICROGLIA_AREA_BINS-1; i++) {
        data_stream << i*MICROGLIA_BIN_AREA << " <= microglial fibre area < " 
                    << (i+1)*MICROGLIA_BIN_AREA << ",";
    }
    data_stream << "microglia fibre area >= " 
                << (NUM_MICROGLIA_AREA_BINS-1)*MICROGLIA_BIN_AREA << ",";

    data_stream << std::endl;
    data_stream.close();

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

