const ROLE_ALIASES = {
  superadmin: 'systemadmin',
  'system admin': 'systemadmin',
  system_admin: 'systemadmin',
  systemadmin: 'systemadmin',
  admin: 'groupadmin',
  groupadmin: 'groupadmin',
  'group admin': 'groupadmin',
  group_admin: 'groupadmin',
};

export const normalizeRoles = (roles) => {
  if (!Array.isArray(roles)) {
    return [];
  }

  const normalized = roles
    .map((role) => (typeof role === 'string' ? role.trim().toLowerCase() : ''))
    .filter(Boolean)
    .map((role) => ROLE_ALIASES[role] || role.replace(/\s+/g, ''));

  return Array.from(new Set(normalized));
};

export const getNormalizedEmail = (user) => {
  if (!user?.email) {
    return '';
  }

  return String(user.email).trim().toLowerCase();
};

export const isSystemAdmin = (user) => {
  if (!user) return false;
  
  const normalizedEmail = getNormalizedEmail(user);
  const roles = normalizeRoles(user.roles || []);

  // Check for system admin in various possible locations
  return (
    user.is_superuser === true ||
    user.is_superuser === 'true' ||
    user.isAdmin === true ||
    user.role === 'systemadmin' ||
    normalizedEmail === 'systemadmin@daybreak.ai' ||
    normalizedEmail === 'superadmin@daybreak.ai' ||
    roles.includes('systemadmin') ||
    (Array.isArray(user.roles) && user.roles.includes('systemadmin'))
  );
};

export const isGroupAdmin = (user) => {
  if (!user) return false;
  
  const roles = normalizeRoles(user.roles || []);
  const normalizedEmail = getNormalizedEmail(user);

  // System admins should be able to manage groups
  if (isSystemAdmin(user)) {
    return true;
  }

  return (
    roles.includes('groupadmin') ||
    normalizedEmail === 'groupadmin@daybreak.ai' ||
    user.role === 'groupadmin' ||
    (Array.isArray(user.roles) && user.roles.includes('groupadmin'))
  );
};

export const getDefaultLandingPath = (user) => {
  if (isSystemAdmin(user)) {
    return '/admin/groups';  // Always send system admins to groups management
  }

  if (isGroupAdmin(user)) {
    return '/games';
  }

  return '/dashboard';
};

const parseRedirectTarget = (target) => {
  if (!target || typeof target !== 'string') {
    return null;
  }

  const trimmed = target.trim();
  if (!trimmed || trimmed.includes('://')) {
    return null;
  }

  let path = trimmed;
  let search = '';
  let hash = '';

  const hashIndex = path.indexOf('#');
  if (hashIndex >= 0) {
    hash = path.slice(hashIndex);
    path = path.slice(0, hashIndex);
  }

  const searchIndex = path.indexOf('?');
  if (searchIndex >= 0) {
    search = path.slice(searchIndex);
    path = path.slice(0, searchIndex);
  }

  if (!path) {
    path = '/';
  }

  if (!path.startsWith('/')) {
    path = `/${path}`;
  }

  if (path.startsWith('//')) {
    return null;
  }

  const normalizedPath = path.length > 1 ? path.replace(/\/+$/, '') : path;

  return {
    pathname: normalizedPath,
    fullPath: `${normalizedPath}${search}${hash}`,
  };
};

export const resolvePostLoginDestination = (user, redirectTo) => {
  const fallback = getDefaultLandingPath(user);
  if (!redirectTo) {
    return fallback;
  }

  const parsed = parseRedirectTarget(redirectTo);
  if (!parsed) {
    return fallback;
  }

  if (isSystemAdmin(user) && (parsed.pathname === '/' || parsed.pathname === '/dashboard')) {
    return fallback;
  }

  return parsed.fullPath;
};

/**
 * Builds the appropriate login URL for redirecting an unauthenticated user.
 * Avoids appending a redirect back to the root route (`/`) so that the
 * application doesn't appear to "recycle" to `/login?redirect=%2F` on first
 * load. Accepts either a React Router location object or a path string.
 */
export const buildLoginRedirectPath = (locationLike) => {
  if (!locationLike) {
    return '/login';
  }

  let pathname = '/';
  let search = '';

  if (typeof locationLike === 'string') {
    const withoutHash = locationLike.split('#')[0] || '/';
    const [pathPart, searchPart] = withoutHash.split('?');
    pathname = pathPart.startsWith('/') ? pathPart : `/${pathPart}`;
    search = searchPart ? `?${searchPart}` : '';
  } else {
    pathname = locationLike.pathname || '/';
    search = locationLike.search || '';
  }

  pathname = pathname.startsWith('/') ? pathname : `/${pathname}`;

  const shouldIncludeRedirect = !(pathname === '/' && !search);

  if (!shouldIncludeRedirect) {
    return '/login';
  }

  const target = `${pathname}${search}`;
  return `/login?redirect=${encodeURIComponent(target)}`;
};
