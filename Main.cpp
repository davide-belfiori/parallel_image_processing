#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <streambuf>
#include <filesystem> 
#include "lyra/lyra.hpp"
#include "CImg.h"
#include "CL/cl.hpp"
#include "FreeImage/FreeImage.h"

using namespace cimg_library;

FIBITMAP* GenericLoader(const char* lpszPathName, FREE_IMAGE_FORMAT &fif, int flag) {

	// check the file signature and deduce its format
	// (the second argument is currently not used by FreeImage)
	fif = FreeImage_GetFileType(lpszPathName, 0);
	if (fif == FIF_UNKNOWN) {
		// no signature ?
		// try to guess the file format from the file extension
		fif = FreeImage_GetFIFFromFilename(lpszPathName);
	}
	// check that the plugin has reading capabilities ...
	if ((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)) {
		// ok, let's load the file
		FIBITMAP *dib = FreeImage_Load(fif, lpszPathName, flag);
		// unless a bad file format, we are done !
		return dib;
	}
	return NULL;
}


float* loadImageData(std::string filename, int& width, int& height) {
	FIBITMAP *dib = NULL;
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;

	dib = GenericLoader(filename.data(), fif, 0);

	if (dib == NULL) {
		std::cout << "Cannot open " << filename.data();
		exit(2);
	}
	else {

		float*output = NULL;

		if (fif != FIF_BMP) {

			dib = FreeImage_ConvertTo32Bits(dib);
			dib = FreeImage_GetChannel(dib, FICC_RED);
		}

		if (dib != NULL) {

			height = FreeImage_GetHeight(dib);
			width = FreeImage_GetWidth(dib);

			output = (float*)malloc(width*height * sizeof(float));

			int pindex = 0;

			for (int y = height - 1; y >= 0; y--) {
				BYTE *bits = (BYTE *)FreeImage_GetScanLine(dib, y);
				for (int x = 0; x < width; x++) {
					
					float val = (float)bits[x];
					output[pindex] = val;

					pindex++;
				}
			}
		}

		return output;
	}

	return NULL;
}

// Display image on the screen
void show_img(CImg<> img, const char* wname = "") {
	CImgDisplay main_disp(img, wname);
	while (!main_disp.is_closed()) {
		main_disp.wait();
	}
}

void fill_zero(float* data, int width, int hieght) {
	for (int i = 0; i < width * hieght; i++) {
		data[i] = 0.0f;
	}
}

std::string read_kernel_code(const char* filename) {
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	if (in)
	{
		std::string contents;
		in.seekg(0, std::ios::end);
		contents.reserve(in.tellg());
		in.seekg(0, std::ios::beg);
		contents.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		in.close();
		return(contents);
	}
	std::cout << "Error reading kernel file, check if " << filename << "exists" << std::endl;
	exit(3);
}

std::string get_rotation_kernel_code() {
	return read_kernel_code("kernel/rot_filter.cl");
}

std::string get_convolution_kernel_code() {
	return read_kernel_code("kernel/conv_filter.cl");
}


// Print information about OpenCL platforms and devices

void query_devices() {
	std::vector<cl::Platform> all_Platform;
	cl::Platform::get(&all_Platform);

	if (all_Platform.size() > 0) {

		int platform_id = 0;
		int device_id = 0;

		std::cout << "Number of Platforms: " << all_Platform.size() << std::endl;

		for (std::vector<cl::Platform>::iterator it = all_Platform.begin(); it != all_Platform.end(); ++it) {
			cl::Platform platform(*it);

			std::cout << "Platform ID: " << platform_id++ << std::endl;
			std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
			std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

			std::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

			for (std::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2) {
				cl::Device device(*it2);

				std::cout << "\tDevice " << device_id++ << ": " << std::endl;
				std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
				std::cout << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
				std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
				std::cout << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
				std::cout << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
				std::cout << "\t\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
				std::cout << "\t\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
				std::cout << "\t\tDevice Max Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
				std::cout << "\t\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
				std::cout << "\t\tDevice Available: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
			}
			std::cout << std::endl;
		}
	}
	else {
		std::cout << "No OpenCL platform available" << std::endl;
	}
}


// Return number of available OprnCL devices

int count_available_devices() {
	std::vector<cl::Platform> all_Platform;
	cl::Platform::get(&all_Platform);

	int count = 0;

	if (all_Platform.size() > 0) {
		for (std::vector<cl::Platform>::iterator it = all_Platform.begin(); it != all_Platform.end(); ++it) {
			
			cl::Platform platform(*it);
			std::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

			for (int i = 0; i < (int) devices.size(); i++) {
				if (devices[i].getInfo< CL_DEVICE_AVAILABLE>()) {
					count++;
				}
			}
		}
	}

	return count;
}


// Check if device with requested ID is avalilable

bool is_device_available(int plat_id, int dev_id) {
	std::vector<cl::Platform> all_Platform;
	cl::Platform::get(&all_Platform);

	if (all_Platform.size() > 0) {
		if (plat_id >= 0 && plat_id < (int) all_Platform.size()) {

			cl::Platform platform = all_Platform[plat_id];
			std::vector<cl::Device> all_devices;
			platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

			if (dev_id >= 0 && dev_id < (int) all_devices.size()) {
				return all_devices[dev_id].getInfo< CL_DEVICE_AVAILABLE>();
			}
		}
	}

	return false;
}


// ROTAZIONE SEQUENZIALE

void rotate(float* image, int width, int height,
	float theta, float* output) {

	float cos_t = cos(theta);
	float sin_t = sin(theta);

	float a = width / 2.0f;
	float b = height / 2.0f;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			// applichimao le equazioni di rotazione
			float target_x = ((float)j - a) * cos_t - ((float)i - b) * sin_t + a;
			float target_y = ((float)j - a) * sin_t + ((float)i - b) * cos_t + b;

			// verifichiamo se le nuove coordinate rientrano nei limiti dell'immagine
			if ((target_x >= 0.0f) && (target_x < (float)width) &&
				(target_y >= 0.0f) && (target_y < (float)height)) {

				// scriviamo il risultato
				output[i * width + j] = image[(int)target_y * width + (int)target_x];
			}
		}
	}
}


// CONVOLUZIONE SEQUENZIALE

float apply_filter(float* image, int width, int height,
	float* filter, int filter_size, int y, int x) {

	float new_pixel = 0.0f;

	int half_filter_size = filter_size / 2;

	int y_coord = 0;
	int x_coord = 0;

	int filterIndex = 0;

	for (int i = -half_filter_size; i <= half_filter_size; i++) {

		y_coord = std::min(height - 1, std::max(y + i, 0));

		for (int j = -half_filter_size; j <= half_filter_size; j++) {

			x_coord = std::min(width - 1, std::max(x + j, 0));
			float srcPx = image[y_coord * width + x_coord];
			new_pixel += srcPx * filter[filterIndex];

			filterIndex++;
		}
	}

	return std::min(255.0f, std::max(new_pixel, 0.0f));
}

void convolution2D(float* image, int width, int height,
	float* filter, int filter_size, float* output) {

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			// calcolo e scrittura del nuovo pixel
			float new_pixel = apply_filter(image, width, height, filter, filter_size, i, j);
			output[i * width + j] = new_pixel;
		}
	}
}


// ROTAZIONE PARALLELA

void p_rotate(float* image, int width, int height, float theta, float* output, int platform_id = 0, int dev_id = 0, bool verbose = true) {

	// otteniamo le piattaforme disponibili sulla macchina
	std::vector<cl::Platform> all_Platform;
	cl::Platform::get(&all_Platform);

	cl::Platform platform;

	if (platform_id < 0 || platform_id >= (int) all_Platform.size()) {
		if (verbose)
			std::cout << "Platform ID " << platform_id << " not found, using default" << std::endl;
		platform = all_Platform[0];
	}
	else {
		if(verbose)
			std::cout << "Using platform ID " << platform_id  << std::endl;
		platform = all_Platform[platform_id];
	}

	// otteniamo tutti i dispositivi della piattaforma
	std::vector<cl::Device> all_devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

	// selezioniamo il primo dispositivo disponibile
	cl::Device dev;

	if (dev_id < 0 || dev_id >= (int) all_devices.size()) {
		if (verbose)
			std::cout << "Device ID " << dev_id << " not found, using default" << std::endl;
		dev = all_devices[0];
	}
	else {
		if (verbose)
			std::cout << "Using device ID " << dev_id << std::endl;
		dev = all_devices[dev_id];
	}

	cl::Context context({ dev });

	cl::Program::Sources sources;
	std::string src_code = get_rotation_kernel_code();
	sources.push_back({ src_code.c_str(), src_code.length() });

	cl::Program program(context, sources);
	program.build({ dev });

	// dichiariamo che l'immagine ha un solo canale 
	// e che i pixel hanno valori di tipo float
	cl::ImageFormat image_format(CL_R, CL_FLOAT);

	// buffer per l'immagine sorgente
	cl::Image2D inputImage(context,
		CL_MEM_READ_ONLY, image_format,
		width, height,
		0, NULL);

	// buffer per l'immagine destinazione
	cl::Image2D outputImage(context,
		CL_MEM_WRITE_ONLY, image_format,
		width, height,
		0, NULL);

	cl::CommandQueue queue(context, dev);

	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	cl::size_t<3> region;
	region[0] = width;
	region[1] = height;
	region[2] = 1;

	queue.enqueueWriteImage(inputImage, CL_TRUE, origin, region,
		width * sizeof(float),
		0, image);

	cl::Kernel kernel(program, "rot_filter");

	kernel.setArg(0, inputImage);
	kernel.setArg(1, outputImage);
	kernel.setArg(2, theta);
	kernel.setArg(3, width);
	kernel.setArg(4, height);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange,
		cl::NDRange(width, height),
		cl::NullRange);
	queue.finish();

	queue.enqueueReadImage(outputImage, CL_TRUE,
		origin, region,
		width * sizeof(float), 0,
		(void*)output);
}


// CONVOLUZIONE PARALLELA

void p_convolution2D(float* image, int width, int height, float* filter, int filter_size, float* output, int platform_id = 0, int dev_id = 0, bool verbose = true) {

	// otteniamo le piattaforme disponibili sulla macchina
	std::vector<cl::Platform> all_Platform;
	cl::Platform::get(&all_Platform);

	cl::Platform platform;

	if (platform_id < 0 || platform_id >= (int) all_Platform.size()) {
		if (verbose)
			std::cout << "Platform ID " << platform_id << " not found, using default" << std::endl;
		platform = all_Platform[0];
	}
	else {
		if (verbose)
			std::cout << "Using platform ID " << platform_id << std::endl;
		platform = all_Platform[platform_id];
	}

	// otteniamo tutti i dispositivi della piattaforma
	std::vector<cl::Device> all_devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

	// selezioniamo il primo dispositivo disponibile
	cl::Device dev;

	if (dev_id < 0 || dev_id >= (int) all_devices.size()) {
		if (verbose)
			std::cout << "Device ID " << dev_id << " not found, using default" << std::endl;
		dev = all_devices[0];
	}
	else {
		if (verbose)
			std::cout << "Using device ID " << dev_id << std::endl;
		dev = all_devices[dev_id];
	}

	cl::Context context({ dev });

	cl::Program::Sources sources;
	std::string src_code = get_convolution_kernel_code();
	sources.push_back({ src_code.c_str(), src_code.length() });

	cl::Program program(context, sources);
	program.build({ dev });

	cl::ImageFormat image_format(CL_R, CL_FLOAT);

	// buffer immagini
	cl::Image2D inputImage(context,
		CL_MEM_READ_ONLY, image_format,
		width, height,
		0, NULL);

	cl::Image2D outputImage(context,
		CL_MEM_WRITE_ONLY, image_format,
		width, height,
		0, NULL);

	// buffer filtro
	cl::Buffer filter_buffer(context, CL_MEM_READ_WRITE,
		sizeof(float) * filter_size*filter_size);

	cl::CommandQueue queue(context, dev);

	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	cl::size_t<3> region;
	region[0] = width;
	region[1] = height;
	region[2] = 1;
	// riempimento buffer immagine sorgente
	queue.enqueueWriteImage(inputImage, CL_TRUE, origin, region,
		width * sizeof(float), 0, image);
	// riempimento buffer filtro
	queue.enqueueWriteBuffer(filter_buffer, CL_TRUE, 0,
		sizeof(float) * filter_size*filter_size,
		filter);

	cl::Kernel kernel(program, "conv_filter");

	kernel.setArg(0, inputImage);
	kernel.setArg(1, outputImage);
	kernel.setArg(2, filter_buffer);
	kernel.setArg(3, filter_size);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange,
		cl::NDRange(width, height),
		cl::NullRange);
	queue.finish();

	queue.enqueueReadImage(outputImage, CL_TRUE,
		origin, region,
		width * sizeof(float), 0,
		(void*)output);
}


float sobel[9] = { -1, 0, 1,
				   -2, 0, 2,
				   -1, 0, 1 };


// BENCHMARK FUNCTIONS

void benchmark_rotation(std::vector<std::string> filenames, int platform_id = 0, int dev_id = 0) {

	std::cout << "ROTATION Benchmark Result (time expressed in seconds) \n\n";
	std::cout << "***************************************************************************\n\n";
	std::cout << "  Image (image size) --> Sequential Execution Time || Parallel Execution Time \n\n";
	std::cout << "***************************************************************************\n\n";

	for (int i = 0; i < (int) filenames.size(); i++) {
		std::string filename = filenames[i];

		int width = 0;
		int height = 0;
		float* image = loadImageData(filename, width, height);

		if (image != NULL) {

			float* output = (float*)malloc(sizeof(float) * width * height);
			fill_zero(output, width, height);

			// Sequential Execution

			auto t1 = std::chrono::high_resolution_clock::now();

			rotate(image, width, height, M_PI_4, output);

			auto t2 = std::chrono::high_resolution_clock::now();
			float exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();

			std::cout << filename << "  (" << width << " x " << height << ") --> " << exec_time;

			// Parallel Execution

			t1 = std::chrono::high_resolution_clock::now();

			p_rotate(image, width, height, M_PI_4, output, platform_id, dev_id, false);

			t2 = std::chrono::high_resolution_clock::now();
			exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();

			std::cout << "  ||  " << exec_time << "\n";
		}
		else {
			std::cerr << "Error in reading " << filename.data() << std::endl;
		}
	}
}

void benchmark_convolution(std::vector<std::string> filenames, int platform_id = 0, int dev_id = 0) {

	std::cout << "CONVOLUTION Benchmark Result (time expressed in seconds) \n\n";
	std::cout << "***************************************************************************\n\n";
	std::cout << "  Image (image size) --> Sequential Execution Time || Parallel Execution Time \n\n";
	std::cout << "***************************************************************************\n\n";

	for (int i = 0; i < (int)filenames.size(); i++) {
		std::string filename = filenames[i];

		int width = 0;
		int height = 0;
		float* image = loadImageData(filename, width, height);

		if (image != NULL) {

			float* output = (float*)malloc(sizeof(float) * width * height);
			fill_zero(output, width, height);

			// Sequential Execution

			auto t1 = std::chrono::high_resolution_clock::now();

			convolution2D(image, width, height, sobel, 3, output);

			auto t2 = std::chrono::high_resolution_clock::now();
			float exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();

			std::cout << filename << "  (" << width << " x " << height << ") --> " << exec_time;

			// Parallel Execution

			t1 = std::chrono::high_resolution_clock::now();

			p_convolution2D(image, width, height, sobel, 3, output, platform_id, dev_id, false);

			t2 = std::chrono::high_resolution_clock::now();
			exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();

			std::cout << "  ||  " << exec_time << "\n";
		}
		else {
			std::cerr << "Error in reading " << filename.data() << std::endl;
		}
	}
}

// CLI COMMAD DEFINITION

struct rotation_cmd {
	std::string rot_target = "";
	float deg = 90.0f;
	bool parallel = false;
	std::string dst_filename = "";
	bool show_help = true;
	int plat_id = 0;
	int dev_id = 0;

	rotation_cmd(lyra::cli &cli) {
		cli.add_argument(
			lyra::command("rotate", [this](const lyra::group & g) { this->run(g); })
			.help("Rotate given image")
			.add_argument(lyra::help(show_help))
			.add_argument(lyra::arg(rot_target, "image").required())
			.add_argument(lyra::opt(deg,"degrees")
				.name("-d")
				.name("--deg")
				.optional()
				.help("Rotation degrees (default = 90°)"))
			.add_argument(lyra::opt(parallel)
				.name("-p")
				.optional()
				.help("Perform parallel rotation"))
			.add_argument(lyra::opt(plat_id, "platform id")
				.name("--plat-id")
				.optional()
				.help("OpenCL platform ID"))
			.add_argument(lyra::opt(dev_id, "device id")
				.name("--dev-id")
				.optional()
				.help("OpenCL device ID"))
			.add_argument(lyra::opt(dst_filename, "path")
				.name("--save")
				.optional()
				.help("Save rotation result"))
		);
	}

	void run(const lyra::group & g) {

		float exec_time = 0.0f;
		int width = 0;
		int height = 0;
		float* image = loadImageData(rot_target, width, height);
		if(image == NULL){
			std::cerr << "Error in reading input file" << std::endl;
			exit(2);
		}

		float* output = (float*)malloc(sizeof(float) * width * height);
		fill_zero(output, width, height);
		float rad = deg * M_PI / 180;

		if (parallel) {

			if (count_available_devices() <= 0) {
				std::cout << "No OpenCL devices available, cannot perform parallel operation" << std::endl;
				exit(11);
			}

			auto t1 = std::chrono::high_resolution_clock::now();

			p_rotate(image, width, height, rad, output, plat_id, dev_id);

			auto t2 = std::chrono::high_resolution_clock::now();
			exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();
		}
		else {
			auto t1 = std::chrono::high_resolution_clock::now();

			rotate(image, width, height, rad, output);

			auto t2 = std::chrono::high_resolution_clock::now();
			exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();
		}

		std::cout << "Image size = " << width << " x " << height << std::endl;
		std::cout << "Execution time: " << exec_time << " s" << std::endl;

		CImg<float> result(output, width, height, 1, 1);
		if (dst_filename != "") {
			result.save(dst_filename.data());
		}
		else {
			show_img(result, "Rotation Result");
		}
	}
};

struct filter_cmd {
	std::string filter_target = "";
	bool parallel = false;
	std::string dst_filename = "";
	bool show_help = true;
	int plat_id = 0;
	int dev_id = 0;

	filter_cmd(lyra::cli &cli) {
		cli.add_argument(
			lyra::command("filter", [this](const lyra::group & g) { this->run(g); })
			.help("Filter given image")
			.add_argument(lyra::help(show_help))
			.add_argument(lyra::arg(filter_target, "image").required())
			.add_argument(lyra::opt(parallel)
				.name("-p")
				.optional()
				.help("Perform parallel filtering"))
			.add_argument(lyra::opt(plat_id, "platform id")
				.name("--plat-id")
				.optional()
				.help("OpenCL platform ID"))
			.add_argument(lyra::opt(dev_id, "device id")
				.name("--dev-id")
				.optional()
				.help("OpenCL device ID"))
			.add_argument(lyra::opt(dst_filename, "path")
				.name("--save")
				.optional()
				.help("Save filter result"))
		);
	}

	void run(const lyra::group & g) {

		float exec_time = 0.0f;
		int width = 0;
		int height = 0;
		float* image = loadImageData(filter_target, width, height);
		if (image == NULL) {
			std::cerr << "Error in reading input file" << std::endl;
			exit(2);
		}

		float* output = (float*)malloc(sizeof(float) * width * height);
		fill_zero(output, width, height);

		if (parallel) {

			if (count_available_devices() <= 0) {
				std::cout << "No OpenCL devices available, cannot perform parallel operation" << std::endl;
				exit(11);
			}

			auto t1 = std::chrono::high_resolution_clock::now();

			p_convolution2D(image, width, height, sobel, 3, output, plat_id, dev_id);

			auto t2 = std::chrono::high_resolution_clock::now();
			exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();
		}
		else {
			auto t1 = std::chrono::high_resolution_clock::now();

			convolution2D(image, width, height, sobel, 3, output);

			auto t2 = std::chrono::high_resolution_clock::now();
			exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();
		}

		std::cout << "Image size = " << width << " x " << height << std::endl;
		std::cout << "Execution time: " << exec_time << " s" <<std::endl;

		CImg<float> result(output, width, height, 1, 1);
		if (dst_filename != "") {
			result.save(dst_filename.data());
		}
		else {
			show_img(result, "Filter Result");
		}
	}
};

struct benchmark_cmd {

	std::vector<std::string> benchmark_images;
	int plat_id = 0;
	int dev_id = 0;

	benchmark_cmd(lyra::cli &cli) {
		cli.add_argument(
			lyra::command("benchmark", [this](const lyra::group & g) { this->run(g); })
			.help("Perform execution time comparison between sequential and parallel approach to rotation and convolution")
			.add_argument(lyra::opt(plat_id, "platform id")
				.name("--plat-id")
				.optional()
				.help("OpenCL platform ID"))
			.add_argument(lyra::opt(dev_id, "device id")
				.name("--dev-id")
				.optional()
				.help("OpenCL device ID"))
			.add_argument(lyra::arg(benchmark_images, "images").required()));
	}

	void run(const lyra::group & g) {

		if (!is_device_available(plat_id, dev_id)) {
			std::cout << "Device ID " << dev_id << " not available for Platform ID " << plat_id << std::endl;
			std::cout << "Please check available devices using 'oclinfo' command" << std::endl;
			exit(10);
		}

		benchmark_rotation(benchmark_images, plat_id, dev_id);
		std::cout << "\n\n";
		benchmark_convolution(benchmark_images, plat_id, dev_id);
	}
};

struct query_cmd {

	query_cmd(lyra::cli &cli) {
		cli.add_argument(
			lyra::command("oclinfo", [this](const lyra::group & g) { this->run(g); })
				.help("Show information about OpenCL platforms and devices available on this machine"));
	}

	void run(const lyra::group & g) {
		query_devices();
	}
};

int main(int argc, const char** args) {

	auto cli = lyra::cli();

	bool show_help = false;
	cli.add_argument(lyra::help(show_help));
	
	rotation_cmd rot_cmd{ cli };
	filter_cmd filter_cmd{ cli };
	benchmark_cmd benchmark_cmd { cli };
	query_cmd q_cmd{ cli };

	auto result = cli.parse({ argc, args });
	if (!result)
	{
		std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
		exit(1);
	}

	if (show_help) {
		std::cout << cli;
	}

	exit(0);

}
