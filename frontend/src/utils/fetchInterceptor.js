// Save the original fetch
const originalFetch = window.fetch;

// Override the global fetch
window.fetch = async function(...args) {
  const [url, options = {}] = args;
  
  // Log the request
  console.log(`[Fetch] ${options.method || 'GET'} ${url}`, {
    body: options.body,
    headers: options.headers
  });
  
  try {
    const response = await originalFetch.apply(this, args);
    
    // Clone the response so we can read it and still return it
    const responseClone = response.clone();
    
    // Log the response
    console.log(`[Fetch Response] ${response.status} ${url}`, {
      status: response.status,
      statusText: response.statusText,
      headers: Object.fromEntries(response.headers.entries()),
      body: await responseClone.text().catch(() => '[could not read body]')
    });
    
    return response;
  } catch (error) {
    console.error('[Fetch Error]', error);
    throw error;
  }
};

export default window.fetch;
