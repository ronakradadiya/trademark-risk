/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['onnxruntime-node'],
  },
  webpack: (config) => {
    // Our TS source uses ESM-style `.js` import specifiers so tsx/Node ESM
    // can run the same files. Tell webpack that `.js` inside source can also
    // resolve to `.ts`/`.tsx` siblings.
    config.resolve.extensionAlias = {
      ...(config.resolve.extensionAlias ?? {}),
      '.js': ['.ts', '.tsx', '.js'],
      '.mjs': ['.mts', '.mjs'],
    };
    return config;
  },
};

export default nextConfig;
