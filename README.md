# Fashion Imagine 👗✨

An AI-powered virtual try-on application that lets you visualize how clothing items would look on you before making a purchase. Upload your photo and a clothing image, and let AI do the magic!

## What It Does

Fashion Imagine uses Google's Gemini 2.5 Flash (via OpenRouter API) to generate realistic virtual try-on images. Simply:

1. **Upload your photo** - A full-body portrait works best
2. **Upload clothing image** - Any clothing item you want to try on
3. **Click "Do the magic"** - AI generates a realistic image of you wearing the clothing
4. **View the result** - See how the outfit looks on you instantly!

## Features

- **3-Column Layout**: Clean, intuitive interface showing your photo, clothing item, and result side-by-side
- **Real-time Preview**: See uploaded images instantly before processing
- **Automatic Image Compression**: Images are automatically optimized to ensure fast processing
- **Beautiful UI**: Glass morphism design with gradient backgrounds and smooth animations
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices

## Live Demo

Visit the live application: [https://fashion-imagine.vercel.app](https://fashion-imagine.vercel.app)

## Tech Stack

- **Frontend**: Next.js 14 (App Router) with TypeScript
- **Styling**: Tailwind CSS v4
- **Animations**: Framer Motion
- **AI Model**: Google Gemini 2.5 Flash Image
- **API**: OpenRouter (proxy to Gemini)
- **Deployment**: Vercel

## Getting Started

### Prerequisites

- Node.js 18+ installed
- An OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sillanaresh/fashion-imagine.git
cd fashion-imagine
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env.local` file in the root directory:
```env
OPENROUTER_API_KEY=your_api_key_here
```

4. Run the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## Best Practices & Tips

### For Best Results:

1. **Photo Quality**:
   - Use well-lit, clear photos
   - Full-body portraits work better than close-ups
   - Avoid busy backgrounds when possible

2. **Clothing Images**:
   - Use clear product images
   - Images with plain backgrounds work best
   - Make sure the clothing item is the main focus

3. **AI Generation**:
   - **If the result isn't perfect, click "Do the magic" again!** AI generation can vary, and you might get better results on subsequent attempts
   - Try 2-3 times if needed - each generation is unique
   - Different prompts or slight image adjustments can improve results

### Technical Limitations:

**Image Size Handling:**
- Vercel serverless functions have a 4.5MB request body limit
- **We automatically handle this!** Images are compressed client-side before upload:
  - Resized to max 1920px on the longest side
  - Compressed to JPEG format with 80% quality
  - This happens automatically - no action needed from you
- The compression maintains visual quality while ensuring compatibility

**Processing Time:**
- Each generation takes 10-30 seconds depending on image complexity
- Be patient and wait for the result to appear

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | Yes |

## Deployment

### Deploy to Vercel

1. Push your code to GitHub
2. Import the project in Vercel
3. Add the `OPENROUTER_API_KEY` environment variable in Vercel project settings
4. Deploy!

The application will automatically redeploy on every push to the main branch.

## Project Structure

```
fashion-imagine/
├── app/
│   ├── api/
│   │   └── try-on/
│   │       └── route.ts      # API endpoint for AI processing
│   ├── globals.css           # Global styles and color scheme
│   ├── layout.tsx            # Root layout
│   └── page.tsx              # Main application page
├── components/
│   └── ImageUploader.tsx     # Image upload component with compression
├── public/                   # Static assets
├── .env.local               # Environment variables (not in repo)
├── next.config.js           # Next.js configuration
├── tailwind.config.ts       # Tailwind CSS configuration
└── package.json             # Dependencies

```

## How It Works

1. **Image Upload**: User uploads two images (personal photo + clothing item)
2. **Client-Side Compression**: Images are automatically compressed using HTML5 Canvas API
3. **API Request**: Compressed images are sent to `/api/try-on` endpoint
4. **AI Processing**: Backend sends images to Gemini 2.5 Flash via OpenRouter
5. **Result Display**: Generated try-on image is displayed in the Result column

## API Usage

The application uses OpenRouter as a proxy to access Google's Gemini 2.5 Flash model:

- **Model**: `google/gemini-2.5-flash-image`
- **Endpoint**: `https://openrouter.ai/api/v1/chat/completions`
- **Request Format**: OpenAI-compatible API with vision capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Troubleshooting

**Issue**: Images won't upload
- **Solution**: Make sure images are in a standard format (JPEG, PNG). They'll be automatically compressed.

**Issue**: "Images are too large" error
- **Solution**: This shouldn't happen with automatic compression, but try using smaller source images if you encounter this.

**Issue**: Generation takes too long
- **Solution**: Vercel functions have a 10-second timeout on free tier. If this is an issue, consider upgrading to Pro tier (60s timeout).

**Issue**: Result doesn't look good
- **Solution**: Click "Do the magic" again! AI generation varies, and you'll likely get a better result on the second or third try.

## Acknowledgments

- Built with [Next.js](https://nextjs.org/)
- AI powered by [Google Gemini](https://deepmind.google/technologies/gemini/)
- API access via [OpenRouter](https://openrouter.ai/)
- Deployed on [Vercel](https://vercel.com/)

---

Made with ❤️ for fashion enthusiasts who want to try before they buy!
