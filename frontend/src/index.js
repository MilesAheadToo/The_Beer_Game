// /frontend/src/index.js
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./index.css";
import App from "./App";
import { AuthProvider } from "./contexts/AuthContext";
import mixedGameApi from "./services/api";
import { API_BASE_URL } from "./config/api.ts";

async function init() {
  console.log("Attempting to connect to API at:", `${API_BASE_URL}/health`);
  try {
    const data = await mixedGameApi.health(); // -> GET http://localhost:8000/api/v1/health
    console.log("API health:", data);
    return true;
  } catch (err) {
    const status = err?.response?.status;
    const text = err?.response?.statusText;
    const payload = err?.response?.data;
    console.error("API connection test failed:", { status, text, payload });
    throw new Error(`API connection failed: HTTP error! status: ${status} - ${text || "Unknown"}`);
  }
}

init()
  .then(() => {
    const root = ReactDOM.createRoot(document.getElementById("root"));
    root.render(
      <BrowserRouter>
        <AuthProvider>
          <App />
        </AuthProvider>
      </BrowserRouter>
    );
  })
  .catch((e) => {
    const el = document.getElementById("root");
    el.innerHTML = `
      <div style="font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 32px;">
        <h1 style="margin:0 0 8px 0;">Error</h1>
        <p style="margin:0 0 16px 0;">Initialization failed: ${e.message}</p>
        <p style="color:#666;margin:0;">Current Step:<br/><strong>Initializing...</strong></p>
        <p style="margin-top:16px;color:#888;">Tip: ensure backend is running at ${API_BASE_URL}</p>
      </div>`;
  });
