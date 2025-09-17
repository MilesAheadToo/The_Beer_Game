import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Button,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  useToast,
  IconButton,
  useDisclosure,
  AlertDialog,
  AlertDialogOverlay,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogBody,
  AlertDialogFooter,
  Input,
  FormControl,
  FormLabel,
  VStack,
  HStack,
  Text,
  Badge,
  useColorModeValue,
  Tooltip
} from '@chakra-ui/react';
import { AddIcon, DeleteIcon, EditIcon, UnlockIcon } from '@chakra-ui/icons';
import { api } from '../services/api';
import { getUserType as resolveUserType } from '../utils/authUtils';

const Users = () => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const tableHeaderBg = useColorModeValue('gray.50', 'gray.700');
  const rowHoverBg = useColorModeValue('gray.50', 'gray.700');

  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [editingUser, setEditingUser] = useState(null);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    is_admin: false
  });
  const toast = useToast();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const cancelRef = React.useRef();

  // const { isAdmin } = useAuth();
  const [isPwOpen, setPwOpen] = useState(false);
  const [pwUser, setPwUser] = useState(null);
  const [newPassword, setNewPassword] = useState('');

  const fetchUsers = useCallback(async () => {
    try {
      const response = await api.get('/users/');
      setUsers(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching users:', error);
      toast({
        title: 'Error',
        description: 'Failed to fetch users',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      setLoading(false);
    }
  }, [toast]);

  const toggleAdmin = async (user) => {
    try {
      const currentType = resolveUserType(user);
      const nextType = currentType === 'systemadmin' ? 'Player' : 'SystemAdmin';
      await api.put(`/users/${user.id}`, { user_type: nextType });
      toast({ title: 'Role updated', status: 'success', duration: 2000, isClosable: true });
      fetchUsers();
    } catch (e) {
      toast({ title: 'Failed to update role', description: e?.response?.data?.detail || e.message, status: 'error', duration: 5000, isClosable: true });
    }
  };

  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? checked : value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const targetType = formData.is_admin ? 'SystemAdmin' : 'Player';
      if (editingUser) {
        // Update: email/full_name only (password uses change-password endpoint)
        await api.put(`/users/${editingUser.id}`, {
          email: formData.email,
          full_name: formData.full_name || undefined,
          user_type: targetType,
        });
        toast({ title: 'User updated', status: 'success', duration: 3000, isClosable: true });
      } else {
        await api.post('/users/', {
          username: formData.username,
          email: formData.email,
          password: formData.password || undefined,
          user_type: targetType,
        });
        toast({ title: 'User created', status: 'success', duration: 3000, isClosable: true });
      }
      handleClose();
      fetchUsers();
    } catch (error) {
      console.error('Error saving user:', error);
      toast({
        title: 'Error',
        description: error.response?.data?.detail || 'Failed to save user',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleEdit = (user) => {
    setEditingUser(user);
    setFormData({
      username: user.username,
      email: user.email,
      password: '',
      is_admin: resolveUserType(user) === 'systemadmin'
    });
    onOpen();
  };

  const handleDelete = async (userId) => {
    try {
      if (!window.confirm('Are you sure you want to delete this user?')) return;
      await api.delete(`/users/${userId}`);
      toast({ title: 'User deleted', status: 'info', duration: 3000, isClosable: true });
      fetchUsers();
    } catch (error) {
      console.error('Error deleting user:', error);
      toast({
        title: 'Error',
        description: 'Failed to delete user',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleClose = () => {
    setEditingUser(null);
    setFormData({
      username: '',
      email: '',
      password: '',
      is_admin: false
    });
    onClose();
  };

  if (loading) {
    return <Box p={4}>Loading users...</Box>;
  }

  return (
    <Box p={4}>
      <VStack spacing={4} align="stretch">
        <HStack justify="space-between" mb={4}>
          <Text fontSize="2xl" fontWeight="bold">User Management</Text>
          <Button
            leftIcon={<AddIcon />}
            colorScheme="blue"
            onClick={() => onOpen()}
          >
            Add User
          </Button>
        </HStack>

        <Box
          borderWidth="1px"
          borderRadius="lg"
          overflow="hidden"
          boxShadow="sm"
          bg={bgColor}
          className="table-surface"
        >
          <Box fontSize="sm" overflowX="auto">
          <Table variant="simple" size="sm">
            <Thead bg={tableHeaderBg}>
              <Tr>
                <Th>Username</Th>
                <Th>Email</Th>
                <Th>Role</Th>
                <Th>Actions</Th>
              </Tr>
            </Thead>
              <Tbody>
                {users.map((user) => (
                  <Tr key={user.id} _hover={{ bg: rowHoverBg }}>
                  <Td>
                    <Text isTruncated maxW={{ base: '12rem', sm: '16rem', md: '20rem' }}>
                      {user.username}
                    </Text>
                  </Td>
                  <Td>
                    <Text isTruncated maxW={{ base: '16rem', sm: '22rem', md: '28rem' }}>
                      {user.email}
                    </Text>
                  </Td>
                 <Td>
                   <HStack>
                      <Badge colorScheme={resolveUserType(user) === 'systemadmin' ? 'purple' : 'green'}>
                        {resolveUserType(user) === 'systemadmin' ? 'Admin' : 'User'}
                      </Badge>
                      <Tooltip label={resolveUserType(user) === 'systemadmin' ? 'Revoke admin' : 'Make admin'}>
                        <input type="checkbox" checked={resolveUserType(user) === 'systemadmin'} onChange={() => toggleAdmin(user)} />
                      </Tooltip>
                    </HStack>
                  </Td>
                  <Td>
                    <HStack spacing={2}>
                      <Tooltip label="Edit user">
                        <IconButton
                          icon={<EditIcon />}
                          size="sm"
                          onClick={() => handleEdit(user)}
                          aria-label="Edit user"
                        />
                      </Tooltip>
                      <Tooltip label="Change password">
                        <IconButton
                          icon={<UnlockIcon />}
                          size="sm"
                          onClick={() => { setPwUser(user); setNewPassword(''); setPwOpen(true); }}
                          aria-label="Change password"
                        />
                      </Tooltip>
                      <Tooltip label="Delete user">
                        <IconButton
                          icon={<DeleteIcon />}
                          size="sm"
                          colorScheme="red"
                          onClick={() => handleDelete(user.id)}
                          aria-label="Delete user"
                        />
                      </Tooltip>
                    </HStack>
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
          </Box>
        </Box>
      </VStack>

      {/* Add/Edit User Modal */}
      <AlertDialog
        isOpen={isOpen}
        leastDestructiveRef={cancelRef}
        onClose={handleClose}
      >
        <AlertDialogOverlay>
          <AlertDialogContent as="form" onSubmit={handleSubmit}>
            <AlertDialogHeader>
              {editingUser ? 'Edit User' : 'Add New User'}
            </AlertDialogHeader>
            <AlertDialogBody>
              <VStack spacing={4}>
                <FormControl isRequired>
                  <FormLabel>Username</FormLabel>
                  <Input
                    name="username"
                    value={formData.username}
                    onChange={handleInputChange}
                    placeholder="Enter username"
                  />
                </FormControl>
                <FormControl isRequired>
                  <FormLabel>Email</FormLabel>
                  <Input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    placeholder="Enter email"
                  />
                </FormControl>
                <FormControl isRequired={!editingUser}>
                  <FormLabel>Password</FormLabel>
                  <Input
                    type="password"
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    placeholder={editingUser ? 'Leave blank to keep current' : 'Enter password'}
                  />
                </FormControl>
                <FormControl display="flex" alignItems="center">
                  <FormLabel mb="0" htmlFor="is_admin">
                    Admin User
                  </FormLabel>
                  <input
                    type="checkbox"
                    id="is_admin"
                    name="is_admin"
                    checked={formData.is_admin}
                    onChange={handleInputChange}
                    style={{ marginLeft: '8px' }}
                  />
                </FormControl>
              </VStack>
            </AlertDialogBody>
            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={handleClose}>
                Cancel
              </Button>
              <Button colorScheme="blue" type="submit" ml={3}>
                {editingUser ? 'Update' : 'Create'}
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>

      {/* Change Password Modal */}
      <AlertDialog isOpen={isPwOpen} leastDestructiveRef={cancelRef} onClose={() => setPwOpen(false)}>
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader>Change Password</AlertDialogHeader>
            <AlertDialogBody>
              <VStack>
                <FormControl isRequired>
                  <FormLabel>New Password</FormLabel>
                  <Input type="password" value={newPassword} onChange={(e)=> setNewPassword(e.target.value)} />
                </FormControl>
              </VStack>
            </AlertDialogBody>
            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={() => setPwOpen(false)}>Cancel</Button>
              <Button colorScheme="blue" ml={3} onClick={async () => {
                try {
                  await api.post(`/users/${pwUser.id}/change-password`, { current_password: '', new_password: newPassword });
                  toast({ title: 'Password updated', status: 'success', duration: 3000, isClosable: true });
                  setPwOpen(false);
                } catch (e) {
                  toast({ title: 'Failed', description: e?.response?.data?.detail || e.message, status: 'error', duration: 5000, isClosable: true });
                }
              }}>Update</Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </Box>
  );
};

export default Users;
