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

  const renderTabContent = () => {
    switch (activeTab) {
      case 0: // Profile
        return (
          <Box>
            <Heading size="md" mb={4}>Profile Information</Heading>
            <VStack spacing={4} align="stretch">
              <HStack spacing={4}>
                <Avatar size="xl" src={profileForm.avatar} />
                <VStack align="start">
                  <Button leftIcon={<FiUpload />} size="sm">Change Photo</Button>
                  <Text fontSize="sm" color="gray.500">JPG, GIF or PNG. Max size 2MB</Text>
                </VStack>
              </HStack>

              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                <FormControl>
                  <FormLabel>First Name</FormLabel>
                  <Input 
                    value={profileForm.firstName}
                    onChange={(e) => setProfileForm({...profileForm, firstName: e.target.value})}
                  />
                </FormControl>
                <FormControl>
                  <FormLabel>Last Name</FormLabel>
                  <Input 
                    value={profileForm.lastName}
                    onChange={(e) => setProfileForm({...profileForm, lastName: e.target.value})}
                  />
                </FormControl>
                <FormControl>
                  <FormLabel>Email</FormLabel>
                  <Input 
                    type="email" 
                    value={profileForm.email}
                    isDisabled
                  />
                </FormControl>
                <FormControl>
                  <FormLabel>Company</FormLabel>
                  <Input 
                    value={profileForm.company}
                    onChange={(e) => setProfileForm({...profileForm, company: e.target.value})}
                  />
                </FormControl>
              </SimpleGrid>

              <FormControl>
                <FormLabel>Bio</FormLabel>
                <Textarea 
                  placeholder="Tell us about yourself..."
                  rows={4}
                />
              </FormControl>

              <HStack justifyContent="flex-end" mt={4}>
                <Button variant="outline" mr={2}>Cancel</Button>
                <Button colorScheme="blue" onClick={handleSaveProfile}>Save Changes</Button>
              </HStack>
            </VStack>
          </Box>
        );

      case 1: // Notifications
        return (
          <Box>
            <Heading size="md" mb={4}>Notification Preferences</Heading>
            <VStack spacing={6} align="stretch">
              <Card variant="outline">
                <CardHeader>
                  <Heading size="md">Email Notifications</Heading>
                </CardHeader>
                <CardBody>
                  <VStack align="stretch" spacing={4}>
                    <HStack justify="space-between">
                      <Box>
                        <Text fontWeight="medium">Account Activity</Text>
                        <Text fontSize="sm" color="gray.500">Receive emails about account activities</Text>
                      </Box>
                      <Switch 
                        isChecked={notificationSettings.email}
                        onChange={handleNotificationChange('email')}
                        colorScheme="blue"
                      />
                    </HStack>
                    <HStack justify="space-between">
                      <Box>
                        <Text fontWeight="medium">Weekly Reports</Text>
                        <Text fontSize="sm" color="gray.500">Get weekly performance reports</Text>
                      </Box>
                      <Switch 
                        isChecked={notificationSettings.weeklyReport}
                        onChange={handleNotificationChange('weeklyReport')}
                        colorScheme="blue"
                      />
                    </HStack>
                  </VStack>
                </CardBody>
              </Card>

              <Card variant="outline">
                <CardHeader>
                  <Heading size="md">Push Notifications</Heading>
                </CardHeader>
                <CardBody>
                  <VStack align="stretch" spacing={4}>
                    <HStack justify="space-between">
                      <Box>
                        <Text fontWeight="medium">Order Updates</Text>
                        <Text fontSize="sm" color="gray.500">Get real-time order updates</Text>
                      </Box>
                      <Switch 
                        isChecked={notificationSettings.push}
                        onChange={handleNotificationChange('push')}
                        colorScheme="blue"
                      />
                    </HStack>
                  </VStack>
                </CardBody>
              </Card>
            </VStack>
          </Box>
        );

      case 2: // API Keys
        return (
          <Box>
            <HStack justify="space-between" mb={6}>
              <Heading size="md">API Keys</Heading>
              <Button 
                leftIcon={<FiPlus />} 
                colorScheme="blue"
                onClick={onOpen}
              >
                Create API Key
              </Button>
            </HStack>

            {apiKeys.length === 0 ? (
              <Box textAlign="center" py={10} borderWidth={1} borderRadius="md" borderStyle="dashed">
                <FiKey size={48} style={{ margin: '0 auto 16px', color: '#718096' }} />
                <Text fontSize="lg" mb={2}>No API Keys</Text>
                <Text color="gray.500" mb={4}>Create your first API key to get started</Text>
                <Button colorScheme="blue" onClick={onOpen}>Create API Key</Button>
              </Box>
            ) : (
              <VStack spacing={4} align="stretch">
                {apiKeys.map((key) => (
                  <Card key={key.id} variant="outline">
                    <CardBody>
                      <HStack justify="space-between">
                        <Box>
                          <Text fontWeight="medium">{key.name}</Text>
                          <Text fontSize="sm" color="gray.500">
                            Created on {new Date(key.created).toLocaleDateString()}
                            {key.lastUsed && ` • Last used ${formatRelativeTime(key.lastUsed)}`}
                          </Text>
                        </Box>
                        <HStack>
                          <Button 
                            size="sm" 
                            variant="outline" 
                            onClick={() => copyToClipboard(key.key)}
                          >
                            Copy Key
                          </Button>
                          <Button 
                            size="sm" 
                            colorScheme="red" 
                            variant="ghost"
                            onClick={() => confirmDeleteKey(key.id)}
                          >
                            <FiTrash2 />
                          </Button>
                        </HStack>
                      </HStack>
                    </CardBody>
                  </Card>
                ))}
              </VStack>
            )}

            <Alert status="info" mt={6} borderRadius="md">
              <AlertIcon />
              <Box>
                <Text fontWeight="bold">Keep your API keys secure</Text>
                <Text fontSize="sm">Do not share them in publicly accessible areas such as GitHub, client-side code, and so forth.</Text>
              </Box>
            </Alert>
          </Box>
        );

      case 3: // Security
        return (
          <Box>
            <Heading size="md" mb={4}>Security Settings</Heading>
            
            <Card variant="outline" mb={6}>
              <CardHeader>
                <Heading size="md">Change Password</Heading>
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
                      />
                      <InputRightElement width="4.5rem">
                        <Button h="1.75rem" size="sm" onClick={() => togglePasswordVisibility('current')}>
                          {showPassword.current ? <FiEyeOff /> : <FiEye />}
                        </Button>
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
                      />
                      <InputRightElement width="4.5rem">
                        <Button h="1.75rem" size="sm" onClick={() => togglePasswordVisibility('new')}>
                          {showPassword.new ? <FiEyeOff /> : <FiEye />}
                        </Button>
                      </InputRightElement>
                    </InputGroup>
                  </FormControl>
                  
                  <FormControl>
                    <FormLabel>Confirm New Password</FormLabel>
                    <InputGroup>
                      <Input
                        type={showPassword.confirm ? 'text' : 'password'}
                        value={passwordForm.confirmPassword}
                        onChange={(e) => setPasswordForm({...passwordForm, confirmPassword: e.target.value})}
                      />
                      <InputRightElement width="4.5rem">
                        <Button h="1.75rem" size="sm" onClick={() => togglePasswordVisibility('confirm')}>
                          {showPassword.confirm ? <FiEyeOff /> : <FiEye />}
                        </Button>
                      </InputRightElement>
                    </InputGroup>
                  </FormControl>
                  
                  <HStack w="full" justify="flex-end" mt={4}>
                    <Button 
                      colorScheme="blue" 
                      onClick={handleChangePassword}
                      isLoading={isSubmitting}
                    >
                      Update Password
                    </Button>
                  </HStack>
                </VStack>
              </CardBody>
            </Card>

            <Card variant="outline">
              <CardHeader>
                <Heading size="md" color="red.500">Danger Zone</Heading>
              </CardHeader>
              <CardBody>
                <Text mb={4}>
                  Permanently delete your account and all associated data. This action cannot be undone.
                </Text>
                <Button colorScheme="red" variant="outline" leftIcon={<FiTrash2 />}>
                  Delete My Account
                </Button>
              </CardBody>
            </Card>
          </Box>
        );

      default:
        return null;
    }
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
        <TabList mb={6} borderBottom="1px" borderColor={borderColor} overflowX="auto" overflowY="hidden">
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
            <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6}>
              <CardHeader borderBottomWidth="1px" borderColor={borderColor}>
                <Text fontSize="lg" fontWeight="semibold">Profile Information</Text>
                <Text fontSize="sm" color="gray.500">Update your account's profile information and email address</Text>
              </CardHeader>
              <CardBody>
                <VStack spacing={4} align="stretch">
                  <FormControl>
                    <FormLabel>Name</FormLabel>
                    <Input placeholder="Your name" value={profile.name} onChange={(e) => setProfile({...profile, name: e.target.value})} />
                  </FormControl>
                  <FormControl>
                    <FormLabel>Email</FormLabel>
                    <Input type="email" placeholder="your.email@example.com" value={profile.email} onChange={(e) => setProfile({...profile, email: e.target.value})} />
                  </FormControl>
                  <FormControl>
                    <FormLabel>Bio</FormLabel>
                    <Textarea placeholder="Tell us about yourself" value={profile.bio} onChange={(e) => setProfile({...profile, bio: e.target.value})} />
                  </FormControl>
                  <HStack justify="flex-end" mt={4}>
                    <Button colorScheme="blue" onClick={handleSaveProfile}>Save Changes</Button>
                  </HStack>
                </VStack>
              </CardBody>
            </Card>
          </TabPanel>
          
          {/* Notifications Tab */}
          <TabPanel px={0}>
            <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6}>
              <CardHeader borderBottomWidth="1px" borderColor={borderColor}>
                <Text fontSize="lg" fontWeight="semibold">Notification Preferences</Text>
                <Text fontSize="sm" color="gray.500">Configure how you receive notifications</Text>
              </CardHeader>
              <CardBody>
                <VStack spacing={4} align="stretch">
                  <FormControl display="flex" alignItems="center" justifyContent="space-between">
                    <Box>
                      <FormLabel mb={0} fontWeight="normal">Email Notifications</FormLabel>
                      <Text fontSize="sm" color="gray.500">Receive notifications via email</Text>
                    </Box>
                    <Switch isChecked={notificationSettings.email} onChange={handleNotificationChange('email')} />
                  </FormControl>
                  
                  <FormControl display="flex" alignItems="center" justifyContent="space-between" pl={8}>
                    <Box>
                      <FormLabel mb={0} fontWeight="normal">Weekly Digest</FormLabel>
                      <Text fontSize="sm" color="gray.500">Get a weekly summary of your activity</Text>
                    </Box>
                    <Switch isChecked={notificationSettings.weeklyReport} onChange={handleNotificationChange('weeklyReport')} isDisabled={!notificationSettings.email} />
                  </FormControl>
                  
                  <FormControl display="flex" alignItems="center" justifyContent="space-between">
                    <Box>
                      <FormLabel mb={0} fontWeight="normal">Push Notifications</FormLabel>
                      <Text fontSize="sm" color="gray.500">Receive push notifications on your device</Text>
                    </Box>
                    <Switch isChecked={notificationSettings.push} onChange={handleNotificationChange('push')} />
                  </FormControl>
                  
                  <FormControl display="flex" alignItems="center" justifyContent="space-between">
                    <Box>
                      <FormLabel mb={0} fontWeight="normal">SMS Alerts</FormLabel>
                      <Text fontSize="sm" color="gray.500">Receive important alerts via SMS</Text>
                    </Box>
                    <Switch isChecked={notificationSettings.sms} onChange={handleNotificationChange('sms')} />
                  </FormControl>
                </VStack>
              </CardBody>
              <CardFooter borderTopWidth="1px" borderColor={borderColor} justify="flex-end">
                <Button colorScheme="blue" onClick={handleSaveNotificationSettings}>Save Preferences</Button>
              </CardFooter>
            </Card>
          </TabPanel>
          
          {/* API Keys Tab */}
          <TabPanel px={0}>
            <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6}>
              <CardHeader borderBottomWidth="1px" borderColor={borderColor}>
                <HStack justify="space-between" align="center">
                  <Box>
                    <Text fontSize="lg" fontWeight="semibold">API Keys</Text>
                    <Text fontSize="sm" color="gray.500">Manage your API access keys</Text>
                  </Box>
                  <Button leftIcon={<FiPlus />} colorScheme="blue" size="sm" onClick={onOpen}>
                    Create New Key
                  </Button>
                </HStack>
              </CardHeader>
              {apiKeys.length === 0 ? (
                <CardBody>
                  <VStack spacing={4} py={8} align="center">
                    <FiKey size={48} color="#A0AEC0" />
                    <Text>No API keys found</Text>
                    <Text fontSize="sm" color="gray.500" textAlign="center" maxW="md">
                      Create your first API key to start integrating with our API
                    </Text>
                    <Button leftIcon={<FiPlus />} colorScheme="blue" mt={4} onClick={onOpen}>
                      Create New Key
                    </Button>
                  </VStack>
                </CardBody>
              ) : (
                <CardBody p={0}>
                  <List spacing={0}>
                    {apiKeys.map((key) => (
                      <ListItem key={key.id} p={4} borderBottomWidth="1px" borderColor={borderColor}>
                        <HStack justify="space-between" w="full">
                          <Box>
                            <Text fontWeight="medium">{key.name}</Text>
                            <HStack spacing={2} color="gray.500" fontSize="sm">
                              <Text>•••••••••{key.key.slice(-4)}</Text>
                              <Text>•</Text>
                              <Text>Created {formatRelativeTime(key.created)}</Text>
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
          </TabPanel>
          
          {/* Security Tab */}
          <TabPanel px={0}>
            <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6}>
              <CardHeader borderBottomWidth="1px" borderColor={borderColor}>
                <Text fontSize="lg" fontWeight="semibold">Change Password</Text>
                <Text fontSize="sm" color="gray.500">Update your account password</Text>
              </CardHeader>
              <CardBody>
                <VStack spacing={4} maxW="md">
                  <FormControl>
                    <FormLabel>Current Password</FormLabel>
                    <InputGroup>
                      <Input
                        type={showPassword.current ? 'text' : 'password'}
                        value={passwordForm.currentPassword}
                        onChange={(e) => setPasswordForm({...passwordForm, currentPassword: e.target.value})}
                      />
                      <InputRightElement width="4.5rem">
                        <Button h="1.75rem" size="sm" onClick={() => togglePasswordVisibility('current')}>
                          {showPassword.current ? <FiEyeOff /> : <FiEye />}
                        </Button>
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
                      />
                      <InputRightElement width="4.5rem">
                        <Button h="1.75rem" size="sm" onClick={() => togglePasswordVisibility('new')}>
                          {showPassword.new ? <FiEyeOff /> : <FiEye />}
                        </Button>
                      </InputRightElement>
                    </InputGroup>
                  </FormControl>
                  
                  <FormControl>
                    <FormLabel>Confirm New Password</FormLabel>
                    <InputGroup>
                      <Input
                        type={showPassword.confirm ? 'text' : 'password'}
                        value={passwordForm.confirmPassword}
                        onChange={(e) => setPasswordForm({...passwordForm, confirmPassword: e.target.value})}
                      />
                      <InputRightElement width="4.5rem">
                        <Button h="1.75rem" size="sm" onClick={() => togglePasswordVisibility('confirm')}>
                          {showPassword.confirm ? <FiEyeOff /> : <FiEye />}
                        </Button>
                      </InputRightElement>
                    </InputGroup>
                  </FormControl>
                  
                  <HStack w="full" justify="flex-end" mt={4}>
                    <Button 
                      colorScheme="blue" 
                      onClick={handleChangePassword}
                      isLoading={isSubmitting}
                    >
                      Update Password
                    </Button>
                  </HStack>
                </VStack>
              </CardBody>
            </Card>

            <Card variant="outline">
              <CardHeader>
                <Heading size="md" color="red.500">Danger Zone</Heading>
              </CardHeader>
              <CardBody>
                <Text mb={4}>
                  Permanently delete your account and all associated data. This action cannot be undone.
                </Text>
                <Button colorScheme="red" variant="outline" leftIcon={<FiTrash2 />}>
                  Delete My Account
                </Button>
              </CardBody>
            </Card>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </PageLayout>
  );
};

export default Settings;
