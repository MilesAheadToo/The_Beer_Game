export const normalizeRoles = (roles) => {
  if (!Array.isArray(roles)) {
    return [];
  }

  return roles
    .map((role) => (typeof role === 'string' ? role.trim().toLowerCase() : ''))
    .filter(Boolean);
};

export const getNormalizedEmail = (user) => {
  if (!user?.email) {
    return '';
  }

  return String(user.email).trim().toLowerCase();
};

export const isSuperAdmin = (user) => {
  const normalizedEmail = getNormalizedEmail(user);
  const roles = normalizeRoles(user?.roles);

  return normalizedEmail === 'superadmin@daybreak.ai' || roles.includes('superadmin');
};

export const isAdmin = (user) => {
  const roles = normalizeRoles(user?.roles);
  const normalizedEmail = getNormalizedEmail(user);

  if (isSuperAdmin(user)) {
    return true;
  }

  return (
    Boolean(user?.is_superuser) ||
    roles.includes('admin') ||
    roles.includes('group_admin') ||
    normalizedEmail === 'admin@daybreak.ai'
  );
};

export const getDefaultLandingPath = (user) => {
  if (isSuperAdmin(user)) {
    return '/admin/groups';
  }

  if (isAdmin(user)) {
    return '/players';
  }

  return '/games';
};
