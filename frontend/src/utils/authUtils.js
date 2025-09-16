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
  const normalizedEmail = getNormalizedEmail(user);
  const roles = normalizeRoles(user?.roles);

  return (
    Boolean(user?.is_superuser) ||
    normalizedEmail === 'systemadmin@daybreak.ai' ||
    normalizedEmail === 'superadmin@daybreak.ai' ||
    roles.includes('systemadmin')
  );
};

export const isGroupAdmin = (user) => {
  const roles = normalizeRoles(user?.roles);
  const normalizedEmail = getNormalizedEmail(user);

  if (isSystemAdmin(user)) {
    return false;
  }

  return (
    roles.includes('groupadmin') ||
    normalizedEmail === 'groupadmin@daybreak.ai'
  );
};

export const getDefaultLandingPath = (user) => {
  if (isSystemAdmin(user)) {
    return '/admin/groups';
  }

  if (isGroupAdmin(user)) {
    return '/games';
  }

  return '/dashboard';
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
