// Save the original fetch
const nativeFetch = window.fetch;

// Override the global fetch
window.fetch = async function(input, init = {}) {
  const url = typeof input === "string" ? input : input.url;
  const options = typeof input === "string" ? init : input;
  const headers = new Headers(options.headers || {});
  const isLogin = /\/api\/v1\/auth\/login\b/.test(url);

  // Log the request
  console.log(`[Fetch] ${options.method || 'GET'} ${url}`, {
    body: options.body,
    headers: Object.fromEntries(headers.entries())
  });
  
  // Attach Authorization for non-login calls
  if (!isLogin) {
    const token = localStorage.getItem("access_token");
    const scheme = (localStorage.getItem("token_type") || "Bearer").replace(/^bearer$/i, "Bearer");
    if (token) headers.set("Authorization", `${scheme} ${token}`);
  }
  
  try {
    const response = await nativeFetch(input, { ...options, headers });
    
    // Log the response
    const responseClone = response.clone();
    console.log(`[Fetch Response] ${response.status} ${url}`, {
      status: response.status,
      statusText: response.statusText,
      headers: Object.fromEntries(response.headers.entries()),
      body: await responseClone.text().catch(() => '[could not read body]')
    });
    
    // Handle 401 Unauthorized
    if (response.status === 401 && !isLogin) {
      const back = encodeURIComponent(window.location.pathname + window.location.search);
      window.location.replace(`/login?redirect=${back}`);
      return new Response(null, { status: 401, statusText: 'Unauthorized' });
    }
    
    return response;
  } catch (error) {
    console.error('[Fetch Error]', error);
    throw error;
  }
};

export default window.fetch;
