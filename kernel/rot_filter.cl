__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void rot_filter(__read_only image2d_t src_img,
							__write_only image2d_t dst_img,
							const float theta,
							const int image_width,
							const int image_height) {

	// coordinate del pixel che il work-item elabora
	int x = get_global_id(0);
	int y = get_global_id(1);

	float cos_t = 0.0;
	float sin_t = sincos(theta, &cos_t);

	float a = image_width / 2;
	float b = image_height / 2;

	float new_x = ((float)x - a) * cos_t - ((float)y - b) * sin_t + a;
	float new_y = ((float)x - a) * sin_t + ((float)y - b) * cos_t + b;

	if ((new_x >= 0.0f) && (new_x < (float)image_width) &&
		(new_y >= 0.0f) && (new_y < (float)image_height)) {

		float4 pxVal = read_imagef(src_img, sampler, (float2)(new_x, new_y));

		write_imagef(dst_img, (int2)(x, y), pxVal);
	}
}
