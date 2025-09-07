import React, { useState } from 'react';
import { 
  Box, 
  VStack, 
  HStack, 
  Text, 
  Button, 
  Input, 
  Textarea, 
  Select, 
  Switch, 
  Divider, 
  Grid, 
  GridItem, 
  Avatar, 
  IconButton, 
  List, 
  ListItem, 
  ListIcon, 
  Card, 
  CardHeader, 
  CardBody, 
  CardFooter, 
  FormControl, 
  FormLabel, 
  FormHelperText,
  useToast,
  useColorModeValue,
  Badge,
  InputGroup,
  InputRightElement,
  InputLeftAddon,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  Heading,
  useBreakpointValue
} from '@chakra-ui/react';
import { 
  FiUser, 
  FiBell, 
  FiLock, 
  FiKey, 
  FiTrash2, 
  FiEdit2, 
  FiPlus, 
  FiSave, 
  FiX,
  FiEye,
  FiEyeOff,
  FiUpload,
  FiCheckCircle,
  FiAlertCircle,
  FiDownload,
  FiMonitor,
  FiSmartphone,
  FiCalendar,
  FiClock
} from 'react-icons/fi';
import PageLayout from '../components/PageLayout';

const Settings = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [activeTab, setActiveTab] = useState(0);
  const toast = useToast();
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const [showPassword, setShowPassword] = useState({
    current: false,
    new: false,
    confirm: false
  });

  // Form states
  const [notificationSettings, setNotificationSettings] = useState({
    email: true,
    push: true,
    weeklyReport: true,
    criticalAlerts: true,
  });

  const [apiKeys, setApiKeys] = useState([
    { 
      id: 'key1', 
      name: 'Production', 
      key: 'sk_test_1234567890', 
      created: '2023-01-15',
      lastUsed: '2023-09-05'
    },
    { 
      id: 'key2', 
      name: 'Development', 
      key: 'sk_test_0987654321', 
      created: '2023-02-20',
      lastUsed: '2023-09-01'
    },
  ]);

  const [newApiKey, setNewApiKey] = useState({ 
    name: '', 
    description: '' 
  });
  
  const [editingKey, setEditingKey] = useState(null);
  
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });

  const [profileForm, setProfileForm] = useState({
    firstName: 'John',
    lastName: 'Doe',
    email: 'john.doe@example.com',
    company: 'Supply Chain Co.',
    position: 'Supply Chain Manager',
    timezone: 'UTC+01:00',
    avatar: 'https://bit.ly/dan-abramov'
  });

  const togglePasswordVisibility = (field) => {
    setShowPassword(prev => ({
      ...prev,
      [field]: !prev[field]
    }));
  };

  const handleNotificationChange = (setting) => (e) => {
    setNotificationSettings({
      ...notificationSettings,
      [setting]: e.target.checked,
    });
  };

  const handleAddApiKey = () => {
    if (newApiKey.name.trim() === '') {
      toast({
        title: 'Error',
        description: 'Please enter a name for the API key',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    
    const key = {
      id: `key_${Date.now()}`,
      name: newApiKey.name,
      description: newApiKey.description,
      key: `sk_test_${Math.random().toString(36).substring(2, 15)}`,
      created: new Date().toISOString().split('T')[0],
      lastUsed: new Date().toISOString().split('T')[0]
    };
    
    setApiKeys([key, ...apiKeys]);
    setNewApiKey({ name: '', description: '' });
    onClose();
    
    toast({
      title: 'API Key Created',
      description: 'Your new API key has been generated successfully.',
      status: 'success',
      duration: 5000,
      isClosable: true,
    });
  };

  const confirmDeleteKey = (id) => {
    if (window.confirm('Are you sure you want to delete this API key? This action cannot be undone.')) {
      setApiKeys(apiKeys.filter(key => key.id !== id));
      toast({
        title: 'API Key Deleted',
        status: 'info',
        duration: 3000,
        isClosable: true,
      });
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    toast({
      title: 'Copied to clipboard',
      status: 'success',
      duration: 2000,
      isClosable: true,
    });
  };

  const handleSaveProfile = async () => {
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      toast({
        title: 'Profile updated',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Error updating profile',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleChangePassword = async () => {
    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      toast({
        title: 'Error',
        description: 'New passwords do not match',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return;
    }
    
    if (passwordForm.newPassword.length < 8) {
      toast({
        title: 'Error',
        description: 'Password must be at least 8 characters long',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return;
    }
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setPasswordForm({
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
      });
      
      toast({
        title: 'Password updated',
        description: 'Your password has been updated successfully.',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Error updating password',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  // Format date to relative time (e.g., "2 days ago")
  const formatRelativeTime = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);
    
    if (diffInSeconds < 60) return 'Just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
    if (diffInSeconds < 2592000) return `${Math.floor(diffInSeconds / 86400)}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <PageLayout title="Account Settings">
      <Tabs 
        variant="enclosed" 
        colorScheme="blue"
        isLazy
        index={activeTab}
        onChange={(index) => setActiveTab(index)}
      >
        <TabList mb={6} borderBottom="1px" borderColor={borderColor}>
          <Tab _selected={{ color: 'blue.600', borderBottom: '2px solid', borderColor: 'blue.500' }}>
            <HStack spacing={2}>
              <FiUser />
              <Text>Profile</Text>
            </HStack>
          </Tab>
          <Tab _selected={{ color: 'blue.600', borderBottom: '2px solid', borderColor: 'blue.500' }}>
            <HStack spacing={2}>
              <FiBell />
              <Text>Notifications</Text>
            </HStack>
          </Tab>
          <Tab _selected={{ color: 'blue.600', borderBottom: '2px solid', borderColor: 'blue.500' }}>
            <HStack spacing={2}>
              <FiKey />
              <Text>API Keys</Text>
              {apiKeys.length > 0 && (
                <Badge colorScheme="blue" variant="solid" borderRadius="full" px={2}>
                  {apiKeys.length}
                </Badge>
              )}
            </HStack>
          </Tab>
          <Tab _selected={{ color: 'blue.600', borderBottom: '2px solid', borderColor: 'blue.500' }}>
            <HStack spacing={2}>
              <FiLock />
              <Text>Security</Text>
            </HStack>
          </Tab>
        </TabList>

        <TabPanels>
          {/* Profile Tab */}
          <TabPanel px={0}>
            <Grid templateColumns={{ base: '1fr', lg: '300px 1fr' }} gap={8}>
              <Box>
                <Card variant="outline" bg={cardBg} borderColor={borderColor}>
                  <CardBody>
                    <VStack spacing={4}>
                      <Avatar 
                        size="2xl" 
                        name={`${profileForm.firstName} ${profileForm.lastName}`} 
                        src={profileForm.avatar}
                        border="2px solid"
                        borderColor="blue.200"
                      />
                      <VStack spacing={1} textAlign="center">
                        <Text fontSize="xl" fontWeight="semibold">
                          {profileForm.firstName} {profileForm.lastName}
                        </Text>
                        <Text color="gray.500">{profileForm.position}</Text>
                        <Text fontSize="sm" color="gray.500">
                          Member since {new Date('2023-01-15').toLocaleDateString('en-US', { year: 'numeric', month: 'long' })}
                        </Text>
                      </VStack>
                      <Button 
                        leftIcon={<FiUpload />} 
                        variant="outline" 
                        size="sm" 
                        width="full"
                        bg="white"
                        color="blue.600"
                        borderColor="blue.600"
                        _hover={{
                          bg: 'blue.50',
                        }}
                        _active={{
                          bg: 'blue.100',
                        }}
                      >
                        Change Photo
                      </Button>
                    </VStack>
                  </CardBody>
                </Card>

                <Card variant="outline" bg={cardBg} borderColor={borderColor} mt={6}>
                  <CardHeader pb={0}>
                    <Text fontWeight="semibold">Login Activity</Text>
                  </CardHeader>
                  <CardBody>
                    <VStack spacing={3} align="stretch">
                      <Box>
                        <Text fontSize="sm" color="gray.500">Last Login</Text>
                        <Text>Today at {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</Text>
                      </Box>
                      <Box>
                        <Text fontSize="sm" color="gray.500">Device</Text>
                        <Text>Chrome on macOS</Text>
                      </Box>
                      <Box>
                        <Text fontSize="sm" color="gray.500">IP Address</Text>
                        <Text>192.168.1.1</Text>
                      </Box>
                    </VStack>
                  </CardBody>
                </Card>
              </Box>

              <Box>
                <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6}>
                  <CardHeader borderBottomWidth="1px" borderColor={borderColor}>
                    <Text fontSize="lg" fontWeight="semibold">Personal Information</Text>
                    <Text fontSize="sm" color="gray.500">Update your personal details here</Text>
                  </CardHeader>
                  <CardBody>
                    <Grid templateColumns={{ base: '1fr', md: '1fr 1fr' }} gap={6}>
                      <FormControl>
                        <FormLabel>First Name</FormLabel>
                        <Input 
                          value={profileForm.firstName}
                          onChange={(e) => setProfileForm({...profileForm, firstName: e.target.value})}
                          placeholder="First name"
                        />
                      </FormControl>
                      
                      <FormControl>
                        <FormLabel>Last Name</FormLabel>
                        <Input 
                          value={profileForm.lastName}
                          onChange={(e) => setProfileForm({...profileForm, lastName: e.target.value})}
                          placeholder="Last name"
                        />
                      </FormControl>
                      
                      <FormControl>
                        <FormLabel>Email Address</FormLabel>
                        <Input 
                          type="email"
                          value={profileForm.email}
                          onChange={(e) => setProfileForm({...profileForm, email: e.target.value})}
                          placeholder="Email address"
                        />
                        <FormHelperText>We'll never share your email.</FormHelperText>
                      </FormControl>
                      
                      <FormControl>
                        <FormLabel>Company</FormLabel>
                        <Input 
                          value={profileForm.company}
                          onChange={(e) => setProfileForm({...profileForm, company: e.target.value})}
                          placeholder="Company name"
                        />
                      </FormControl>
                      
                      <FormControl>
                        <FormLabel>Position</FormLabel>
                        <Input 
                          value={profileForm.position}
                          onChange={(e) => setProfileForm({...profileForm, position: e.target.value})}
                          placeholder="Your position"
                        />
                      </FormControl>
                      
                      <FormControl>
                        <FormLabel>Timezone</FormLabel>
                        <Select 
                          value={profileForm.timezone}
                          onChange={(e) => setProfileForm({...profileForm, timezone: e.target.value})}
                          placeholder="Select timezone"
                        >
                          <option value="UTC+00:00">UTC±00:00 (GMT)</option>
                          <option value="UTC+01:00">UTC+01:00 (CET)</option>
                          <option value="UTC-05:00">UTC-05:00 (EST)</option>
                          <option value="UTC-08:00">UTC-08:00 (PST)</option>
                        </Select>
                      </FormControl>
                      
                      <FormControl gridColumn={{ base: '1 / -1', md: '1 / 2' }}>
                        <FormLabel>Bio</FormLabel>
                        <Textarea 
                          placeholder="Tell us about yourself..."
                          rows={4}
                        />
                        <FormHelperText>Brief description for your profile.</FormHelperText>
                      </FormControl>
                    </Grid>
                  </CardBody>
                  <CardFooter borderTopWidth="1px" borderColor={borderColor} justifyContent="flex-end">
                    <Button 
                      bg="blue.600"
                      color="white"
                      leftIcon={<FiSave />}
                      onClick={handleSaveProfile}
                      _hover={{
                        bg: 'blue.700',
                      }}
                      _active={{
                        bg: 'blue.800',
                      }}
                    >
                      Save Changes
                    </Button>
                  </CardFooter>
                </Card>

                {/* Change Password Card */}
                <Card variant="outline" bg={cardBg} borderColor={borderColor}>
                  <CardHeader borderBottomWidth="1px" borderColor={borderColor}>
                    <Text fontSize="lg" fontWeight="semibold">Change Password</Text>
                    <Text fontSize="sm" color="gray.500">Update your password regularly to keep your account secure</Text>
                  </CardHeader>
                  <CardBody>
                    <VStack spacing={4}>
                      <FormControl>
                        <FormLabel>Current Password</FormLabel>
                        <InputGroup>
                          <Input
                            type={showPassword.current ? 'text' : 'password'}
                            value={passwordForm.currentPassword}
                            onChange={(e) => setPasswordForm({...passwordForm, currentPassword: e.target.value})}
                            placeholder="Enter current password"
                          />
                          <InputRightElement>
                            <IconButton
                              icon={showPassword.current ? <FiEyeOff /> : <FiEye />}
                              variant="ghost"
                              size="sm"
                              onClick={() => togglePasswordVisibility('current')}
                              aria-label={showPassword.current ? 'Hide password' : 'Show password'}
                            />
                          </InputRightElement>
                        </InputGroup>
                      </FormControl>
                      
                      <FormControl>
                        <FormLabel>New Password</FormLabel>
                        <InputGroup>
                          <Input
                            type={showPassword.new ? 'text' : 'password'}
                            value={passwordForm.newPassword}
                            onChange={(e) => setPasswordForm({...passwordForm, newPassword: e.target.value})}
                            placeholder="Enter new password"
                          />
                          <InputRightElement>
                            <IconButton
                              icon={showPassword.new ? <FiEyeOff /> : <FiEye />}
                              variant="ghost"
                              size="sm"
                              onClick={() => togglePasswordVisibility('new')}
                              aria-label={showPassword.new ? 'Hide password' : 'Show password'}
                            />
                          </InputRightElement>
                        </InputGroup>
                        <FormHelperText>Must be at least 8 characters long</FormHelperText>
                      </FormControl>
                      
                      <FormControl>
                        <FormLabel>Confirm New Password</FormLabel>
                        <InputGroup>
                          <Input
                            type={showPassword.confirm ? 'text' : 'password'}
                            value={passwordForm.confirmPassword}
                            onChange={(e) => setPasswordForm({...passwordForm, confirmPassword: e.target.value})}
                            placeholder="Confirm new password"
                            isInvalid={passwordForm.newPassword !== '' && 
                                      passwordForm.confirmPassword !== '' && 
                                      passwordForm.newPassword !== passwordForm.confirmPassword}
                          />
                          <InputRightElement>
                            <IconButton
                              icon={showPassword.confirm ? <FiEyeOff /> : <FiEye />}
                              variant="ghost"
                              size="sm"
                              onClick={() => togglePasswordVisibility('confirm')}
                              aria-label={showPassword.confirm ? 'Hide password' : 'Show password'}
                            />
                          </InputRightElement>
                        </InputGroup>
                        {passwordForm.newPassword !== '' && 
                         passwordForm.confirmPassword !== '' && 
                         passwordForm.newPassword !== passwordForm.confirmPassword && (
                          <FormHelperText color="red.500">
                            Passwords do not match
                          </FormHelperText>
                        )}
                      </FormControl>
                    </VStack>
                  </CardBody>
                  <CardFooter borderTopWidth="1px" borderColor={borderColor} justifyContent="flex-end">
                    <Button 
                      bg="blue.600"
                      color="white"
                      leftIcon={<FiSave />}
                      isDisabled={
                        !passwordForm.currentPassword ||
                        !passwordForm.newPassword ||
                        passwordForm.newPassword !== passwordForm.confirmPassword ||
                        passwordForm.newPassword.length < 8
                      }
                      onClick={handleChangePassword}
                      _hover={{
                        bg: 'blue.700',
                      }}
                      _active={{
                        bg: 'blue.800',
                      }}
                      _disabled={{
                        bg: 'gray.200',
                        color: 'gray.500',
                        cursor: 'not-allowed',
                      }}
                    >
                      Update Password
                    </Button>
                  </CardFooter>
                </Card>
              </Box>
            </Grid>
          </TabPanel>
          
          {/* Notifications Tab */}
          <TabPanel px={0}>
            <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6}>
              <CardHeader borderBottomWidth="1px" borderColor={borderColor}>
                <Text fontSize="lg" fontWeight="semibold">Email Notifications</Text>
                <Text fontSize="sm" color="gray.500">Manage your email notification preferences</Text>
              </CardHeader>
              <CardBody>
                <VStack spacing={6} align="stretch">
                  <FormControl display="flex" alignItems="center" justifyContent="space-between">
                    <Box>
                      <Text fontWeight="medium">Account Notifications</Text>
                      <Text fontSize="sm" color="gray.500">Important updates about your account</Text>
                    </Box>
                    <Switch 
                      colorScheme="blue"
                      isChecked={notificationSettings.email}
                      onChange={handleNotificationChange('email')}
                    />
                  </FormControl>
                  
                  <FormControl display="flex" alignItems="center" justifyContent="space-between">
                    <Box>
                      <Text fontWeight="medium">Push Notifications</Text>
                      <Text fontSize="sm" color="gray.500">Receive notifications in your browser</Text>
                    </Box>
                    <Switch 
                      colorScheme="blue"
                      isChecked={notificationSettings.push}
                      onChange={handleNotificationChange('push')}
                    />
                  </FormControl>
                  
                  <FormControl display="flex" alignItems="center" justifyContent="space-between">
                    <Box>
                      <Text fontWeight="medium">Weekly Reports</Text>
                      <Text fontSize="sm" color="gray.500">Get a weekly summary of your activity</Text>
                    </Box>
                    <Switch 
                      colorScheme="blue"
                      isChecked={notificationSettings.weeklyReport}
                      onChange={handleNotificationChange('weeklyReport')}
                    />
                  </FormControl>
                  
                  <FormControl display="flex" alignItems="center" justifyContent="space-between">
                    <Box>
                      <Text fontWeight="medium">Critical Alerts</Text>
                      <Text fontSize="sm" color="gray.500">Immediate notifications for important events</Text>
                    </Box>
                    <Switch 
                      colorScheme="blue"
                      isChecked={notificationSettings.criticalAlerts}
                      onChange={handleNotificationChange('criticalAlerts')}
                    />
                  </FormControl>
                </VStack>
              </CardBody>
              <CardFooter borderTopWidth="1px" borderColor={borderColor} justifyContent="flex-end">
                <Button 
                  bg="blue.600"
                  color="white"
                  _hover={{
                    bg: 'blue.700',
                  }}
                  _active={{
                    bg: 'blue.800',
                  }}
                >
                  Save Preferences
                </Button>
              </CardFooter>
            </Card>
            
            <Alert status="info" variant="left-accent" borderRadius="md" mb={6}>
              <AlertIcon />
              <Box>
                <AlertTitle>Notification Settings</AlertTitle>
                <AlertDescription fontSize="sm">
                  Some notifications are required and cannot be turned off, such as important security alerts.
                </AlertDescription>
              </Box>
            </Alert>
          </TabPanel>
          
          {/* API Keys Tab */}
          <TabPanel px={0}>
            <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6}>
              <CardHeader 
                borderBottomWidth="1px" 
                borderColor={borderColor}
                display="flex"
                justifyContent="space-between"
                alignItems="center"
              >
                <Box>
                  <Text fontSize="lg" fontWeight="semibold">API Keys</Text>
                  <Text fontSize="sm" color="gray.500">Manage your API access keys</Text>
                </Box>
                <Button 
                  leftIcon={<FiPlus />} 
                  bg="blue.600"
                  color="white"
                  size="sm"
                  onClick={onOpen}
                  _hover={{
                    bg: 'blue.700',
                  }}
                  _active={{
                    bg: 'blue.800',
                  }}
                >
                  Create API Key
                </Button>
              </CardHeader>
              
              {apiKeys.length === 0 ? (
                <CardBody textAlign="center" py={10}>
                  <FiKey size={32} style={{ margin: '0 auto 16px', color: '#718096' }} />
                  <Text fontWeight="medium" mb={2}>No API keys found</Text>
                  <Text color="gray.500" mb={6} maxW="md" mx="auto">
                    You don't have any API keys yet. Create your first API key to get started with the API.
                  </Text>
                  <Button 
                    leftIcon={<FiPlus />} 
                    bg="blue.600"
                    color="white"
                    onClick={onOpen}
                    _hover={{
                      bg: 'blue.700',
                    }}
                    _active={{
                      bg: 'blue.800',
                    }}
                  >
                    Create API Key
                  </Button>
                </CardBody>
              ) : (
                <CardBody p={0}>
                  <List spacing={0}>
                    {apiKeys.map((key) => (
                      <ListItem 
                        key={key.id} 
                        borderBottomWidth="1px" 
                        borderColor={borderColor}
                        _last={{ borderBottom: 'none' }}
                        p={4}
                      >
                        <Box flex="1">
                          <HStack spacing={3} mb={2}>
                            <Text fontWeight="medium">{key.name}</Text>
                            <Badge colorScheme={key.name === 'Production' ? 'green' : 'blue'} variant="subtle" size="sm">
                              {key.name}
                            </Badge>
                          </HStack>
                          <HStack spacing={4} fontSize="sm" color="gray.500">
                            <HStack spacing={1}>
                              <FiKey size={14} />
                              <Text>sk_...{key.key.slice(-4)}</Text>
                            </HStack>
                            <HStack spacing={1}>
                              <FiCalendar size={14} />
                              <Text>Created {formatRelativeTime(key.created)}</Text>
                            </HStack>
                            <HStack spacing={1}>
                              <FiClock size={14} />
                              <Text>Last used {formatRelativeTime(key.lastUsed)}</Text>
                            </HStack>
                          </HStack>
                        </Box>
                        <HStack spacing={2}>
                          <IconButton
                            icon={<FiCopy />}
                            variant="ghost"
                            size="sm"
                            aria-label="Copy API key"
                            onClick={() => copyToClipboard(key.key)}
                          />
                          <IconButton
                            icon={<FiTrash2 />}
                            variant="ghost"
                            size="sm"
                            colorScheme="red"
                            aria-label="Delete API key"
                            onClick={() => confirmDeleteKey(key.id)}
                          />
                        </HStack>
                      </ListItem>
                    ))}
                  </List>
                </CardBody>
              )}
            </Card>
            
            <Alert status="warning" variant="left-accent" borderRadius="md">
              <AlertIcon />
              <Box>
                <AlertTitle>Keep your API keys secure</AlertTitle>
                <AlertDescription fontSize="sm">
                  Do not share your API keys in client-side code or expose them in public repositories.
                  For security, we only show the first few characters of your API key after creation.
                </AlertDescription>
              </Box>
            </Alert>
            
            {/* Create API Key Modal */}
            <Modal isOpen={isOpen} onClose={onClose} size="lg">
              <ModalOverlay />
              <ModalContent>
                <ModalHeader>Create New API Key</ModalHeader>
                <ModalCloseButton />
                <ModalBody pb={6}>
                  <VStack spacing={4}>
                    <FormControl isRequired>
                      <FormLabel>Name</FormLabel>
                      <Input 
                        placeholder="e.g., Production Server" 
                        value={newApiKey.name}
                        onChange={(e) => setNewApiKey({...newApiKey, name: e.target.value})}
                      />
                      <FormHelperText>Choose a name that helps you identify this key's purpose.</FormHelperText>
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Description (Optional)</FormLabel>
                      <Textarea 
                        placeholder="What's this key for?"
                        value={newApiKey.description}
                        onChange={(e) => setNewApiKey({...newApiKey, description: e.target.value})}
                        rows={3}
                      />
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Permissions</FormLabel>
                      <VStack align="stretch" spacing={2}>
                        <HStack justify="space-between">
                          <Box>
                            <Text fontWeight="medium">Read Access</Text>
                            <Text fontSize="sm" color="gray.500">View resources</Text>
                          </Box>
                          <Switch defaultChecked colorScheme="blue" />
                        </HStack>
                        <HStack justify="space-between">
                          <Box>
                            <Text fontWeight="medium">Write Access</Text>
                            <Text fontSize="sm" color="gray.500">Create and update resources</Text>
                          </Box>
                          <Switch defaultChecked colorScheme="blue" />
                        </HStack>
                        <HStack justify="space-between">
                          <Box>
                            <Text fontWeight="medium">Admin Access</Text>
                            <Text fontSize="sm" color="gray.500">Full access including deletion</Text>
                          </Box>
                          <Switch colorScheme="blue" />
                        </HStack>
                      </VStack>
                    </FormControl>
                  </VStack>
                </ModalBody>

                <ModalFooter borderTopWidth="1px" borderColor={borderColor}>
                  <Button 
                    variant="outline" 
                    mr={3} 
                    onClick={onClose}
                    borderColor="gray.300"
                    _hover={{
                      bg: 'gray.50',
                    }}
                  >
                    Cancel
                  </Button>
                  <Button 
                    bg="blue.600" 
                    color="white"
                    leftIcon={<FiKey />}
                    onClick={handleAddApiKey}
                    isDisabled={!newApiKey.name.trim()}
                    _hover={{
                      bg: 'blue.700',
                    }}
                    _active={{
                      bg: 'blue.800',
                    }}
                    _disabled={{
                      bg: 'gray.200',
                      color: 'gray.500',
                      cursor: 'not-allowed',
                    }}
                  >
                    Create API Key
                  </Button>
                </ModalFooter>
              </ModalContent>
            </Modal>
          </TabPanel>
          
          {/* Security Tab */}
          <TabPanel px={0}>
            <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6}>
              <CardHeader borderBottomWidth="1px" borderColor={borderColor}>
                <Text fontSize="lg" fontWeight="semibold">Two-Factor Authentication</Text>
                <Text fontSize="sm" color="gray.500">Add an extra layer of security to your account</Text>
              </CardHeader>
              <CardBody>
                <HStack justify="space-between" align="center">
                  <Box>
                    <Text fontWeight="medium">Two-Factor Authentication</Text>
                    <Text fontSize="sm" color="gray.500">Require a verification code when signing in</Text>
                  </Box>
                  <Button 
                    variant="outline" 
                    size="sm"
                    color="blue.600"
                    borderColor="blue.600"
                    _hover={{
                      bg: 'blue.50',
                    }}
                  >
                    Enable 2FA
                  </Button>
                </HStack>
              </CardBody>
            </Card>
            
            <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6}>
              <CardHeader borderBottomWidth="1px" borderColor={borderColor}>
                <Text fontSize="lg" fontWeight="semibold">Active Sessions</Text>
                <Text fontSize="sm" color="gray.500">Manage your logged-in devices</Text>
              </CardHeader>
              <CardBody p={0}>
                <List spacing={0}>
                  <ListItem p={4} borderBottomWidth="1px" borderColor={borderColor}>
                    <HStack spacing={4}>
                      <Box p={2} bg="blue.50" borderRadius="md" color="blue.500">
                        <FiMonitor size={24} />
                      </Box>
                      <Box flex="1">
                        <Text fontWeight="medium">macOS on Chrome</Text>
                        <Text fontSize="sm" color="gray.500">San Francisco, CA, USA • {formatRelativeTime(Date.now() - 3600000)}</Text>
                      </Box>
                      <Badge colorScheme="green" variant="subtle">Current Session</Badge>
                    </HStack>
                  </ListItem>
                  <ListItem p={4}>
                    <HStack spacing={4}>
                      <Box p={2} bg="gray.100" borderRadius="md" color="gray.500">
                        <FiSmartphone size={24} />
                      </Box>
                      <Box flex="1">
                        <Text fontWeight="medium">iOS on Mobile Safari</Text>
                        <Text fontSize="sm" color="gray.500">New York, NY, USA • {formatRelativeTime(Date.now() - 86400000 * 2)}</Text>
                      </Box>
                      <Button 
                        size="sm" 
                        variant="outline" 
                        colorScheme="red"
                        _hover={{
                          bg: 'red.50',
                        }}
                      >
                        Sign Out
                      </Button>
                    </HStack>
                  </ListItem>
                </List>
              </CardBody>
              <CardFooter borderTopWidth="1px" borderColor={borderColor}>
                <Button 
                  variant="link" 
                  color="blue.600" 
                  size="sm"
                  _hover={{
                    textDecoration: 'none',
                    color: 'blue.700',
                  }}
                >
                  Sign out of all other devices
                </Button>
              </CardFooter>
            </Card>
            
            <Card variant="outline" bg={useColorModeValue('red.50', 'red.900')} borderColor={useColorModeValue('red.200', 'red.800')}>
              <CardHeader borderBottomWidth="1px" borderColor={useColorModeValue('red.200', 'red.800')}>
                <Text fontSize="lg" fontWeight="semibold" color={useColorModeValue('red.700', 'red.200')}>
                  Danger Zone
                </Text>
                <Text fontSize="sm" color={useColorModeValue('red.600', 'red.300')}>
                  These actions are irreversible. Please be certain.
                </Text>
              </CardHeader>
              <CardBody>
                <VStack spacing={4} align="stretch">
                  <HStack justify="space-between" align="center">
                    <Box>
                      <Text fontWeight="medium">Delete Account</Text>
                      <Text fontSize="sm" color={useColorModeValue('red.600', 'red.300')}>
                        Permanently delete your account and all associated data
                      </Text>
                    </Box>
                    <Button 
                      variant="outline" 
                      size="sm"
                      color="red.600"
                      borderColor="red.600"
                      _hover={{
                        bg: 'red.50',
                      }}
                    >
                      Delete Account
                    </Button>
                  </HStack>
                  
                  <HStack justify="space-between" align="center">
                    <Box>
                      <Text fontWeight="medium">Export Data</Text>
                      <Text fontSize="sm" color={useColorModeValue('red.600', 'red.300')}>
                        Download all your data in a ZIP file
                      </Text>
                    </Box>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      leftIcon={<FiDownload />}
                      color="gray.600"
                      borderColor="gray.300"
                      _hover={{
                        bg: 'gray.50',
                      }}
                    >
                      Export Data
                    </Button>
                  </HStack>
                </VStack>
              </CardBody>
            </Card>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </PageLayout>

            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Email Notifications
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.email}
                    onChange={handleNotificationChange('email')}
                    color="primary"
                  />
                }
                label="Enable email notifications"
                sx={{ mb: 1, display: 'block' }}
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.weeklyReport}
                    onChange={handleNotificationChange('weeklyReport')}
                    color="primary"
                    disabled={!notificationSettings.email}
                  />
                }
                label="Weekly summary report"
                sx={{ mb: 1, display: 'block', ml: 4 }}
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.criticalAlerts}
                    onChange={handleNotificationChange('criticalAlerts')}
                    color="primary"
                    disabled={!notificationSettings.email}
                  />
                }
                label="Critical alerts"
                sx={{ display: 'block', ml: 4 }}
              />
            </Paper>

            <Paper sx={{ p: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Push Notifications
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.push}
                    onChange={handleNotificationChange('push')}
                    color="primary"
                  />
                }
                label="Enable push notifications"
              />
            </Paper>
          </Box>
        );

      case 2: // API Keys
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              API Keys
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              Manage your API keys for programmatic access to the Supply Chain API.
            </Typography>

            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <TextField
                label="Key Name"
                value={newApiKey.name}
                onChange={(e) => setNewApiKey({ ...newApiKey, name: e.target.value })}
                size="small"
                sx={{ flex: 1 }}
                placeholder="e.g., Production, Development"
              />
              {editingKey ? (
                <>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<SaveIcon />}
                    onClick={handleUpdateApiKey}
                    disabled={!newApiKey.name.trim()}
                  >
                    Update
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<CancelIcon />}
                    onClick={handleCancelEdit}
                  >
                    Cancel
                  </Button>
                </>
              ) : (
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<AddIcon />}
                  onClick={handleAddApiKey}
                  disabled={!newApiKey.name.trim()}
                >
                  Add Key
                </Button>
              )}
            </Box>

            <Paper>
              <List>
                <ListSubheader>Your API Keys</ListSubheader>
                {apiKeys.length === 0 ? (
                  <ListItem>
                    <ListItemText primary="No API keys found" />
                  </ListItem>
                ) : (
                  apiKeys.map((key) => (
                    <ListItem key={key.id} divider>
                      <ListItemText
                        primary={key.name}
                        secondary={`Created: ${key.created} • ${key.key}`}
                        sx={{ wordBreak: 'break-all' }}
                      />
                      <ListItemSecondaryAction>
                        <IconButton
                          edge="end"
                          onClick={() => handleEditApiKey(key)}
                          sx={{ mr: 1 }}
                        >
                          <EditIcon />
                        </IconButton>
                        <IconButton
                          edge="end"
                          onClick={() => handleDeleteApiKey(key.id)}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))
                )}
              </List>
            </Paper>

            <Alert severity="info" sx={{ mt: 3 }}>
              <strong>Keep your API keys secure.</strong> Do not share them in publicly accessible
              areas such as GitHub, client-side code, and so forth.
            </Alert>
          </Box>
        );

      case 3: // Security
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Security Settings
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              Manage your account security settings and active sessions.
            </Typography>

            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Two-Factor Authentication
              </Typography>
              <Typography variant="body2" paragraph>
                Add an extra layer of security to your account by enabling two-factor authentication.
              </Typography>
              <Button variant="outlined" color="primary">
                Set Up Two-Factor Authentication
              </Button>
            </Paper>

            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Active Sessions
              </Typography>
              <Typography variant="body2" paragraph>
                This is a list of devices that have logged into your account. Revoke any sessions that you do not recognize.
              </Typography>
              <List>
                <ListItem divider>
                  <ListItemText
                    primary="Chrome on Windows 10"
                    secondary={`Current session • Last active: ${new Date().toLocaleString()}`}
                  />
                  <ListItemSecondaryAction>
                    <Button color="error" size="small">
                      Revoke
                    </Button>
                  </ListItemSecondaryAction>
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Firefox on macOS"
                    secondary={`Last active: 2 days ago`}
                  />
                  <ListItemSecondaryAction>
                    <Button color="error" size="small">
                      Revoke
                    </Button>
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </Paper>

            <Paper sx={{ p: 3 }}>
              <Typography variant="subtitle1" gutterBottom color="error">
                Danger Zone
              </Typography>
              <Typography variant="body2" paragraph>
                Permanently delete your account and all associated data. This action cannot be undone.
              </Typography>
              <Button variant="outlined" color="error" startIcon={<DeleteIcon />}>
                Delete My Account
              </Button>
            </Paper>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      
      <Tabs
        value={tabValue}
        onChange={handleTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ mb: 3 }}
      >
        <Tab icon={<PersonIcon />} label="Profile" />
        <Tab icon={<NotificationsIcon />} label="Notifications" />
        <Tab icon={<ApiIcon />} label="API Keys" />
        <Tab icon={<SecurityIcon />} label="Security" />
      </Tabs>

      {renderTabContent()}

      <Snackbar
        open={showKeyAlert}
        autoHideDuration={6000}
        onClose={() => setShowKeyAlert(false)}
        message="New API key created. Make sure to copy it now as you won't be able to see it again!"
        action={
          <Button color="secondary" size="small" onClick={() => setShowKeyAlert(false)}>
            OK
          </Button>
        }
      />
    </Box>
  );
};

export default Settings;
