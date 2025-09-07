const BASE_URL = "http://localhost:8000/api/v1";

export async function login({ username, password, grant_type = "password" }) {
  const body = new URLSearchParams({ username, password, grant_type });

  const res = await fetch(`${BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Login failed with ${res.status}`);
  }

  const data = await res.json();
  // Return tokens; AuthContext will store them.
  return {
    access_token: data.access_token,
    token_type: data.token_type,
    refresh_token: data.refresh_token,
  };
}

// Export other auth-related functions as needed
export default {
  login
};
