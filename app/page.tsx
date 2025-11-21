'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import ImageUploader from '@/components/ImageUploader';

export default function Home() {
  const [clothingImage, setClothingImage] = useState<string | null>(null);
  const [userImage, setUserImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTryOn = async () => {
    if (!clothingImage || !userImage) {
      setError('Please upload both images');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResultImage(null);

    try {
      const response = await fetch('/api/try-on', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          clothingImage,
          userImage,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to process images');
      }

      setResultImage(data.resultImage);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col">
      <header className="px-6 md:px-10 lg:px-16 py-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center"
            style={{ paddingLeft: '32px', gap: '12px' }}
          >
            <h1 className="text-xl font-semibold tracking-tight text-gray-900">
              Fashion Imagine
            </h1>
            <div className="flex items-center" style={{ gap: '8px', paddingTop: '2px' }}>
              <span style={{ fontSize: '24px' }}>👗</span>
              <span style={{ fontSize: '24px' }}>👔</span>
            </div>
          </motion.div>
        </div>
      </header>

      <section className="px-6 md:px-10 lg:px-16 pt-16 pb-8">
        <div className="max-w-6xl mx-auto text-center">
          <motion.h2
            initial={{ opacity: 0, y: 22 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
            className="text-4xl md:text-5xl font-semibold text-gray-900 tracking-tight"
          >
            See how it looks on you
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.08 }}
            className="mt-4 text-base md:text-lg text-gray-600"
          >
            Upload your photo and a clothing image to get generated result.
          </motion.p>
        </div>
      </section>

      <section className="px-8 md:px-12 lg:px-20 pb-16">
        <div className="max-w-[1400px] mx-auto flex justify-center items-center" style={{ gap: '48px' }}>
          {/* Your Photo Card */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="card-surface p-8 text-center flex flex-col"
            style={{ width: '360px' }}
          >
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Your Photo</h3>
            <p className="text-sm text-gray-600 mb-6">Drag your portrait or click to upload.</p>

            <div className="flex-1 mb-6" style={{ paddingLeft: '16px', paddingRight: '16px', minHeight: '320px' }}>
              <ImageUploader
                label=""
                sublabel=""
                image={userImage}
                onImageChange={setUserImage}
                icon=""
              />
            </div>

            <p className="text-gray-500 leading-relaxed" style={{ fontSize: '11px' }}>
              Try to upload a full body portrait<br />for better results
            </p>
          </motion.div>

          {/* Plus symbol */}
          <div className="text-gray-400 text-3xl font-light">+</div>

          {/* Clothing Image Card */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="card-surface p-8 text-center flex flex-col"
            style={{ width: '360px' }}
          >
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Clothing Image</h3>
            <p className="text-sm text-gray-600 mb-6">Add the product photo you want to try.</p>

            <div className="flex-1 mb-6" style={{ paddingLeft: '16px', paddingRight: '16px', minHeight: '320px' }}>
              <ImageUploader
                label=""
                sublabel=""
                image={clothingImage}
                onImageChange={setClothingImage}
                icon=""
              />
            </div>

            <p className="text-gray-500 leading-relaxed" style={{ fontSize: '11px' }}>
              Try to upload a clean & clear image<br />of desired clothing
            </p>
          </motion.div>

          {/* Arrow symbol */}
          <div className="text-gray-400 text-3xl font-light">→</div>

          {/* Result Card */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="card-surface p-8 text-center flex flex-col"
            style={{ width: '360px' }}
          >
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Result</h3>
            <p className="text-sm text-gray-600 mb-6">Preview the generated try-on instantly.</p>

            <div className="flex-1 mb-6" style={{ paddingLeft: '16px', paddingRight: '16px', minHeight: '320px' }}>
              <div className="w-full rounded-[1.4rem] border border-[#dfe3f4] bg-gradient-to-b from-[#f6f7ff] to-[#ecefff] flex items-center justify-center" style={{ height: '100%', minHeight: '320px' }}>
                {resultImage ? (
                  <div className="relative w-full h-full">
                    <img
                      src={resultImage}
                      alt="Result"
                      className="h-full w-full object-cover rounded-[1.4rem]"
                    />
                  </div>
                ) : (
                  <div className="text-center px-8">
                    <p className="text-sm leading-relaxed text-gray-500">
                      Your result will appear<br />here once generated.
                    </p>
                  </div>
                )}
              </div>
            </div>

            <motion.button
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              onClick={handleTryOn}
              disabled={isProcessing || !clothingImage || !userImage}
              className="gradient-button rounded-full font-semibold tracking-wide disabled:cursor-not-allowed disabled:opacity-50"
              style={{ width: '60%', height: '29px', fontSize: '12.6px', color: 'white', margin: '0 auto' }}
            >
              {isProcessing ? 'Processing...' : 'Do the magic'}
            </motion.button>

            <p className="text-gray-500 leading-relaxed mt-4" style={{ fontSize: '11px' }}>
              Please try a couple of times if desired<br />results are not achieved. Image models are probabilistic
            </p>

            {resultImage && (
              <button
                onClick={() => setResultImage(null)}
                className="mt-3 text-sm text-gray-500 hover:text-gray-700 transition-colors"
              >
                Reset
              </button>
            )}
            {error && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-4 text-sm text-red-500"
              >
                {error}
              </motion.p>
            )}
          </motion.div>
        </div>
      </section>

      <footer className="px-6 md:px-10 lg:px-16 pb-12 mt-auto">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-sm text-gray-500">All images are processed privately</p>
        </div>
      </footer>
    </main>
  );
}
