__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void conv_filter(__read_only image2d_t src_img,
						  __write_only image2d_t dst_img,
						  __global const float* filter,
						  const int filter_size) {

	int x = get_global_id(0);
	int y = get_global_id(1);

	float4 newPx = (float4) (0.0f, 0.0f, 0.0f, 1.0f);

	int half_filter_size = (int) filter_size / 2;

	int filterIndex = 0;

	int2 coord = (int2) (0,0);

	for (int i = -half_filter_size; i <= half_filter_size; i++) {
		coord.y = y + i;
		for (int j = -half_filter_size; j <= half_filter_size; j++) {
			coord.x = x + j;
			float4 srcPx = read_imagef(src_img, sampler, coord);
			newPx.x += srcPx.x * filter[filterIndex];
			filterIndex++;
		}
	}

	if (newPx.x < 0.0f) { newPx.x = 0.0f; };
	if (newPx.x > 255.0f) { newPx.x = 255.0f; };

	write_imagef(dst_img, (int2)(x, y), newPx);
}