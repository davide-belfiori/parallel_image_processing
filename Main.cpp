#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <fstream>
#include <streambuf>
#include <chrono>
#include "CImg.h"
#include "CL/cl.hpp"

using namespace cimg_library;

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
	throw(errno);
}

std::string get_rotation_kernel_code() {
	return read_kernel_code("kernel/rot_filter.cl");
}

std::string get_convolution_kernel_code() {
	return read_kernel_code("kernel/conv_filter.cl");
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

		y_coord = std::min(height, std::max(y + i, 0));

		for (int j = -half_filter_size; j <= half_filter_size; j++) {

			x_coord = std::min(width, std::max(x + j, 0));
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

void p_rotate(float* image, int width, int height, float theta, float* output) {

	// otteniamo le piattaforme disponibili sulla macchina
	std::vector<cl::Platform> all_Platform;
	cl::Platform::get(&all_Platform);

	// la piattaforma in posizione 0 e' quella relativa alla GPU
	cl::Platform platform = all_Platform[0];

	// otteniamo tutti i dispositivi della piattaforma
	std::vector<cl::Device> all_devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);

	// selezioniamo il primo dispositivo disponibile
	cl::Device dev = all_devices[0];

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

void p_convolution2D(float* image, int width, int height, float* filter, int filter_size, float* output) {

	// otteniamo le piattaforme disponibili sulla macchina
	std::vector<cl::Platform> all_Platform;
	cl::Platform::get(&all_Platform);

	// la piattaforma in posizione 0 e' quella relativa alla GPU
	cl::Platform platform = all_Platform[0];

	// otteniamo tutti i dispositivi della piattaforma
	std::vector<cl::Device> all_devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);

	// selezioniamo il primo dispositivo disponibile
	cl::Device dev = all_devices[0];

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

std::string file_prefix = "image/sample_";
std::vector<std::string> img_size = { "100", "200", "300", "450", "600", "800", "1000", "2000" };
std::string ext = ".bmp";

float sobel[9] = { -1, 0, 1,
				   -2, 0, 2,
				   -1, 0, 1 };

void benchmark_rotation() {

	std::cout << "ROTATION Benchmark  Result (time expressed in seconds) \n\n";
	std::cout << "***************************************************************************\n\n";
	std::cout << "  Image Dimensions --> Sequential Execution Time || Parallel Execution Time \n\n";
	std::cout << "***************************************************************************\n\n";

	for (int i = 0; i < img_size.size(); i++) {
		std::string filename = file_prefix + img_size[i] + ext;
		CImg<float> image(filename.data());

		float* output = (float*)malloc(sizeof(float) * image.width() * image.height());
		fill_zero(output, image.width(), image.height());

		// Sequential Execution

		auto t1 = std::chrono::high_resolution_clock::now();

		rotate(image.data(), image.width(), image.height(), M_PI_4, output);

		auto t2 = std::chrono::high_resolution_clock::now();
		float exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();

		std::cout << "  (" << image.width() << " x " << image.height() << ") --> " << exec_time;

		// Parallel Execution

		t1 = std::chrono::high_resolution_clock::now();

		p_rotate(image.data(), image.width(), image.height(), M_PI_4, output);

		t2 = std::chrono::high_resolution_clock::now();
		exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();

		std::cout << "  ||  " << exec_time << "\n";
	}
}

void benchmark_convolution() {

	std::cout << "CONVOLUTION Benchmark Result (time expressed in seconds) \n\n";
	std::cout << "***************************************************************************\n\n";
	std::cout << "  Image Dimensions --> Sequential Execution Time || Parallel Execution Time \n\n";
	std::cout << "***************************************************************************\n\n";

	for (int i = 0; i < img_size.size(); i++) {
		std::string filename = file_prefix + img_size[i] + ext;
		CImg<float> image(filename.data());

		float* output = (float*)malloc(sizeof(float) * image.width() * image.height());
		fill_zero(output, image.width(), image.height());

		// Sequential Execution

		auto t1 = std::chrono::high_resolution_clock::now();

		convolution2D(image.data(), image.width(), image.height(), sobel, 3, output);

		auto t2 = std::chrono::high_resolution_clock::now();
		float exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();

		std::cout << "  (" << image.width() << " x " << image.height() << ") --> " << exec_time;

		// Parallel Execution

		t1 = std::chrono::high_resolution_clock::now();

		p_convolution2D(image.data(), image.width(), image.height(), sobel, 3, output);

		t2 = std::chrono::high_resolution_clock::now();
		exec_time = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1).count();

		std::cout << "  ||  " << exec_time << "\n";
	}
}

int main() {

	benchmark_rotation();
	std::cout << "\n\n";
	benchmark_convolution();
}