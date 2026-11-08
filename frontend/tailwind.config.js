/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Dark meeting-app palette (Teams-like, not Microsoft branding)
        shell: { 950: "#0f0f14", 900: "#16161d", 850: "#1c1c25", 800: "#23232e", 700: "#2e2e3b", 600: "#3d3d4d" },
        brand: { 300: "#a5a8f0", 400: "#8b8ee8", 500: "#7073d6", 600: "#5b5fc7", 700: "#4a4db0" },
      },
      fontFamily: {
        sans: ['"Segoe UI"', "system-ui", "-apple-system", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
      },
      keyframes: {
        "toast-in": { from: { opacity: 0, transform: "translateY(8px)" }, to: { opacity: 1, transform: "none" } },
        pulseRing: { "0%,100%": { boxShadow: "0 0 0 0 rgba(112,115,214,.6)" }, "50%": { boxShadow: "0 0 0 4px rgba(112,115,214,0)" } },
      },
      animation: { "toast-in": "toast-in .25s ease-out", "pulse-ring": "pulseRing 1.6s ease-in-out infinite" },
    },
  },
  plugins: [],
};
