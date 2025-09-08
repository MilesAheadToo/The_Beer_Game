// /frontend/src/App.js
import React from "react";
import { Routes, Route, Navigate, Outlet, useLocation } from "react-router-dom";
import { Box, CircularProgress } from "@mui/material";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import MixedGamesList from "./pages/MixedGamesList";
import CreateMixedGame from "./pages/CreateMixedGame";
import GameBoard from "./pages/GameBoard";
import Login from "./pages/Login";
import { WebSocketProvider } from "./contexts/WebSocketContext";
import { useAuth } from "./contexts/AuthContext";       // <— unified
import "./utils/fetchInterceptor";

window.onerror = function (message, source, lineno, colno, error) {
  console.error("Global error:", { message, source, lineno, colno, error });
  return false;
};
window.onunhandledrejection = function (event) {
  console.error("Unhandled rejection (promise):", event.reason);
};

function RequireAuth() {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  if (!isAuthenticated) {
    const back = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/login?redirect=${back}`} replace />;
  }

  return <Outlet />;
}

const AppContent = () => {
  const location = useLocation();
  const isGamePage = location.pathname.startsWith("/games/");

  return (
    <Box sx={{ display: "flex" }}>
      <Box component="main" sx={{ flexGrow: 1, p: 3, width: "100%" }}>
        <Routes>
          <Route path="/login" element={<Login />} />

          <Route element={<RequireAuth />}>
            <Route
              path="/dashboard"
              element={
                <>
                  <Navbar />
                  <Dashboard />
                </>
              }
            />

            <Route
              path="/games"
              element={
                <>
                  <Navbar />
                  <MixedGamesList />
                </>
              }
            />

            <Route
              path="/games/new"
              element={
                <>
                  <Navbar />
                  <CreateMixedGame />
                </>
              }
            />

            <Route
              path="/games/:gameId"
              element={
                isGamePage ? (
                  <WebSocketProvider>
                    <Navbar />
                    <GameBoard />
                  </WebSocketProvider>
                ) : (
                  <>
                    <Navbar />
                    <GameBoard />
                  </>
                )
              }
            />

            <Route path="/" element={<Navigate to="/games" replace />} />
            <Route path="*" element={<Navigate to="/games" replace />} />
          </Route>
        </Routes>
      </Box>
    </Box>
  );
};

export default AppContent;
