#pragma once
#include "texture_buffer.cuh"
namespace KittlesPT
{
	class MipChain
	{
	public:

		void init()
		{
			for (uint32_t mip_level = 0; mip_level < max_mip_count; mip_level++)
			{
				mip_textures.push_back(TextureBuffer());
			}
		}

		void resize(int32_t base_width, int32_t base_height)
		{
			max_mip_level = getMaxValidMipLevels({ base_width, base_height });
			max_mip_level = std::min(max_mip_level, max_mip_count - 1);

			for (uint32_t miplevel = 0; miplevel <= max_mip_level; miplevel++)
			{
				int32_t mip_width = base_width >> miplevel;
				int32_t	mip_height = base_height >> miplevel;

				TextureBuffer& mip_texture = mip_textures[miplevel];

				if (mip_texture.isInitialised()) {
					mip_texture.resize(mip_width, mip_height);
				}
				else {
					mip_texture.initialize(mip_width, mip_height);
				}
			}
		}

		void destroy()
		{
			for (TextureBuffer& tex : mip_textures)
			{
				tex.destroy();
			}
			mip_textures.clear();
		}

		//excludes mip0
		static int32_t getMaxValidMipLevels(int2 t_base_resolution)
		{
			int32_t mipx = static_cast<int32_t>(std::log2(t_base_resolution.x)),
				mipy = static_cast<int32_t>(std::log2(t_base_resolution.y));
			return std::min(mipx, mipy);
		}

	public:
		const uint32_t max_mip_count = 7;
		uint32_t max_mip_level = 0;
		std::vector<TextureBuffer> mip_textures;
	};
}