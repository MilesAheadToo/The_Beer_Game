import { createTheme } from "@mui/material/styles";

const PRIMARY = "#16A34A";     // Luma-like green (tailwind green-600)
const PRIMARY_HOVER = "#15803D";
const ACCENT = "#0EA5E9";      // optional accent (teal/blue)
const BG_DEFAULT = "#F8FAFC";  // light gray seen in screenshots
const BG_PAPER = "#FFFFFF";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: PRIMARY, dark: PRIMARY_HOVER, contrastText: "#fff" },
    secondary: { main: ACCENT },
    background: { default: BG_DEFAULT, paper: BG_PAPER },
    text: { primary: "#0F172A", secondary: "#475569" },
    divider: "rgba(15, 23, 42, 0.08)",
  },
  typography: {
    fontFamily: `Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, "Noto Sans", "Helvetica Neue", "Apple Color Emoji","Segoe UI Emoji"`,
    h1: { fontWeight: 700, fontSize: "2.25rem", letterSpacing: "-0.01em" },
    h2: { fontWeight: 700, fontSize: "1.875rem" },
    h3: { fontWeight: 600, fontSize: "1.5rem" },
    button: { fontWeight: 600, textTransform: "none", letterSpacing: 0 },
  },
  shape: { borderRadius: 16 },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        "html, body, #root": { height: "100%" },
        body: { backgroundColor: BG_DEFAULT },
        main: { backgroundColor: BG_DEFAULT },
      },
    },
    // Pill, shadowed "Contact Us" style buttons everywhere
    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: {
        root: {
          borderRadius: 9999,
          paddingInline: 20,
          paddingBlock: 10,
          boxShadow:
            "0 1px 1px rgba(0,0,0,.04), 0 8px 16px -8px rgba(22,163,74,.35)",
        },
        containedPrimary: {
          ":hover": {
            backgroundColor: PRIMARY_HOVER,
            boxShadow:
              "0 1px 1px rgba(0,0,0,.04), 0 10px 18px -10px rgba(22,163,74,.45)",
          },
        },
      },
    },
    // Glassy top bar like Luma; we then pad pages to create the gap
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: "rgba(255,255,255,0.8)",
          backdropFilter: "blur(8px)",
          color: "#0F172A",
          boxShadow: "inset 0 -1px 0 rgba(15,23,42,0.06)",
        },
      },
    },
    MuiToolbar: {
      styleOverrides: { root: { minHeight: 72 } },
    },
    // Soft card surface for shaded blocks / role selection
    MuiPaper: {
      defaultProps: { elevation: 0, variant: "outlined" },
      styleOverrides: {
        outlined: {
          background: "#FFFFFF",
          borderColor: "rgba(15,23,42,0.08)",
          boxShadow: "0 1px 2px rgba(15,23,42,0.06)",
        },
      },
    },
  },
});

export default theme;
