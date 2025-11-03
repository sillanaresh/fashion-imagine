'use client';

import { useRef } from 'react';
import { motion } from 'framer-motion';
import Image from 'next/image';

interface ImageUploaderProps {
  label: string;
  sublabel: string;
  image: string | null;
  onImageChange: (image: string | null) => void;
  icon: string;
}

export default function ImageUploader({
  label,
  sublabel,
  image,
  onImageChange,
  icon,
}: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        onImageChange(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation();
    onImageChange(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div onClick={handleClick} className="relative w-full cursor-pointer group">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />

      <div className="w-full rounded-[1.4rem] border border-[#dfe3f4] bg-gradient-to-b from-[#fafbff] to-[#eef1ff] shadow-[0_16px_32px_rgba(120,132,170,0.12)] transition-transform duration-300 group-hover:-translate-y-0.5 overflow-hidden" style={{ height: '100%', minHeight: '320px', position: 'relative' }}>
        {image ? (
          <>
            <img
              src={image}
              alt="Uploaded"
              className="w-full h-full object-cover"
            />
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleRemove}
              className="absolute top-3 right-3 flex h-9 w-9 items-center justify-center rounded-full bg-white/90 text-gray-600 shadow-[0_10px_18px_rgba(120,132,170,0.16)] transition-colors hover:bg-white z-10"
            >
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </motion.button>
          </>
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center text-sm leading-relaxed text-gray-500">
              Click to upload or drag a file
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
