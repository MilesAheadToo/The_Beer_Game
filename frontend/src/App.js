import React from "react";
import { Routes, Route, Navigate, Outlet, useLocation } from "react-router-dom";
import { Box, CircularProgress } from "@mui/material";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import MixedGamesList from "./pages/MixedGamesList";
import CreateMixedGame from "./pages/CreateMixedGame";
import CreateGameFromConfig from "./components/game/CreateGameFromConfig";
import GameBoard from "./pages/GameBoard";
import Login from "./pages/Login";
import { WebSocketProvider } from "./contexts/WebSocketContext";
import { useAuth } from "./contexts/AuthContext";
import "./utils/fetchInterceptor";
import ProtectedRoute from "./components/ProtectedRoute";
import AdminDashboard from "./pages/admin/Dashboard.jsx";
import AdminTraining from "./pages/admin/Training.jsx";
import ModelSetup from "./pages/admin/ModelSetup.jsx";
import Users from "./pages/Users";
import Settings from "./pages/Settings";
import SystemConfig from "./pages/SystemConfig.jsx";
import Unauthorized from "./pages/Unauthorized";
import SupplyChainConfigList from "./components/supply-chain-config/SupplyChainConfigList";
import SupplyChainConfigForm from "./components/supply-chain-config/SupplyChainConfigForm";
import Players from "./pages/Players.jsx";
import DebugBanner from "./components/DebugBanner.jsx";

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
      <DebugBanner />
      <Box component="main" sx={{ flexGrow: 1, px: 3, py: 0, width: "100%" }}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/unauthorized" element={<Unauthorized />} />

          <Route element={<RequireAuth />}>
            <Route
              path="/dashboard"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <Dashboard />
                </>
              }
            />

            <Route
              path="/games"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <MixedGamesList />
                </>
              }
            />

            <Route
              path="/games/new"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <CreateMixedGame />
                </>
              }
            />

            <Route
              path="/games/new-from-config/:configId"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <CreateGameFromConfig />
                </>
              }
            />

            <Route
              path="/games/:gameId"
              element={
                isGamePage ? (
                  <WebSocketProvider>
                    <Navbar />
                    <Box sx={(theme) => theme.mixins.toolbar} />
                    <GameBoard />
                  </WebSocketProvider>
                ) : (
                  <>
                    <Navbar />
                    <Box sx={(theme) => theme.mixins.toolbar} />
                    <GameBoard />
                  </>
                )
              }
            />

            {/* Admin routes */}
            <Route
              path="/admin"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <AdminDashboard />
                </>
              }
            />
            <Route
              path="/admin/training"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <AdminTraining />
                </>
              }
            />
            <Route
              path="/admin/model-setup"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <ModelSetup />
                </>
              }
            />
            <Route
              path="/users"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <Users />
                </>
              }
            />

            <Route
              path="/settings"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <Settings />
                </>
              }
            />

            <Route
              path="/system-config"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <SystemConfig />
                </>
              }
            />

            <Route
              path="/players"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <Players />
                </>
              }
            />

            <Route
              path="/supply-chain-config"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <SupplyChainConfigList />
                </>
              }
            />
            <Route
              path="/supply-chain-config/new"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <SupplyChainConfigForm />
                </>
              }
            />
            <Route
              path="/supply-chain-config/edit/:id"
              element={
                <>
                  <Navbar />
                  <Box sx={(theme) => theme.mixins.toolbar} />
                  <SupplyChainConfigForm />
                </>
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
