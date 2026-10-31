/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eef9ff',
          100: '#d9f0ff',
          500: '#0f766e',
          600: '#0b5d5b',
          700: '#0d3d4c',
        },
      },
      boxShadow: {
        soft: '0 10px 30px rgba(15, 118, 110, 0.12)',
      },
    },
  },
  plugins: [],
};
