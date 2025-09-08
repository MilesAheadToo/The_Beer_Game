import React, { useState, useEffect } from 'react';
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
import { AddIcon, DeleteIcon, EditIcon } from '@chakra-ui/icons';
import { api } from '../services/api';
import { useAuth } from '../contexts/AuthContext';

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

  const { isAdmin } = useAuth();

  const fetchUsers = async () => {
    try {
      const response = await api.get('/auth/users/');
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
  };

  useEffect(() => {
    fetchUsers();
  }, []);

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
      // Disable create/update in this build; backend endpoints not exposed
      toast({
        title: 'Not available',
        description: 'User create/update is disabled in this build.',
        status: 'info',
        duration: 4000,
        isClosable: true,
      });
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
      is_admin: user.is_superuser || (Array.isArray(user.roles) && user.roles.includes('admin'))
    });
    onOpen();
  };

  const handleDelete = async (userId) => {
    try {
      // Disable delete in this build; backend endpoint not exposed
      toast({
        title: 'Not available',
        description: 'User deletion is disabled in this build.',
        status: 'info',
        duration: 4000,
        isClosable: true,
      });
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
        >
          <Box fontSize="sm">
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
                  <Td>{user.username}</Td>
                  <Td>{user.email}</Td>
                  <Td>
                    <Badge colorScheme={(user.is_superuser || (Array.isArray(user.roles) && user.roles.includes('admin'))) ? 'purple' : 'green'}>
                      {(user.is_superuser || (Array.isArray(user.roles) && user.roles.includes('admin'))) ? 'Admin' : 'User'}
                    </Badge>
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
    </Box>
  );
};

export default Users;
