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
    return '/system-config';
  }

  if (isGroupAdmin(user)) {
    return '/games';
  }

  return '/dashboard';
};
