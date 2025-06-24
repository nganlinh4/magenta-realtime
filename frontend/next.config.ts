import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow cross-origin requests from network IP
  allowedDevOrigins: [
    'localhost:3000',
    '127.0.0.1:3000',
    '192.168.0.4:3000',
    '0.0.0.0:3000'
  ],

  // Enable experimental features for better development
  experimental: {
    turbo: {
      // Enable turbopack for faster builds
    }
  }
};

export default nextConfig;
