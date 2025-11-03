'use client';

import { motion } from 'framer-motion';
import Image from 'next/image';
import { useState } from 'react';

interface ResultDisplayProps {
  originalImage: string;
  resultImage: string;
  onReset: () => void;
}

export default function ResultDisplay({
  originalImage,
  resultImage,
  onReset,
}: ResultDisplayProps) {
  const [isComparing, setIsComparing] = useState(false);

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = resultImage;
    link.download = 'fashion-imagine-result.png';
    link.click();
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="max-w-6xl mx-auto"
    >
      <div className="text-center mb-8">
        <h2 className="text-4xl font-bold mb-4">
          <span className="gradient-text">Your Virtual Try-On</span>
        </h2>
        <p className="text-gray-400">
          {isComparing ? 'Comparing before and after' : 'Here\'s how it looks on you'}
        </p>
      </div>

      <div className="glass-effect rounded-2xl p-8 mb-8">
        {isComparing ? (
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <p className="text-sm text-gray-400 mb-3 text-center">Before</p>
              <div className="relative aspect-[3/4] rounded-xl overflow-hidden">
                <Image
                  src={originalImage}
                  alt="Original"
                  fill
                  className="object-contain"
                />
              </div>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-3 text-center">After</p>
              <div className="relative aspect-[3/4] rounded-xl overflow-hidden">
                <Image
                  src={resultImage}
                  alt="Result"
                  fill
                  className="object-contain"
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="relative aspect-[3/4] max-w-2xl mx-auto rounded-xl overflow-hidden">
            <Image
              src={resultImage}
              alt="Result"
              fill
              className="object-contain"
            />
          </div>
        )}
      </div>

      <div className="flex flex-wrap justify-center gap-4">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setIsComparing(!isComparing)}
          className="px-8 py-4 glass-effect rounded-full text-white font-semibold hover:bg-white/10 transition-all"
        >
          {isComparing ? 'Hide Comparison' : 'Compare Before/After'}
        </motion.button>

        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleDownload}
          className="px-8 py-4 bg-gradient-to-r from-[#00f5ff] to-[#ff006e] text-white font-semibold rounded-full shadow-lg shadow-[#00f5ff]/20 hover:shadow-[#00f5ff]/40 transition-all"
        >
          Download Image
        </motion.button>

        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onReset}
          className="px-8 py-4 glass-effect rounded-full text-white font-semibold hover:bg-white/10 transition-all"
        >
          Try Another Outfit
        </motion.button>
      </div>
    </motion.div>
  );
}
