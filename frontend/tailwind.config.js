/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          50: "#eff6ff",
          100: "#dbeafe",
          200: "#bfdbfe",
          400: "#60a5fa",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
        },
        surface: {
          DEFAULT: "#131314",
          card: "#1e1f20",
          hover: "#282a2c",
          border: "#3c4043",
        },
      },
      fontFamily: {
        sans: ['"Google Sans"', '"Segoe UI"', "Roboto", "sans-serif"],
      },
    },
  },
  plugins: [],
};
