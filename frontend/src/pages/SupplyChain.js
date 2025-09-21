import React, { useState, useCallback } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
  Typography,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Factory as FactoryIcon,
  Store as StoreIcon,
  LocalShipping as ShippingIcon,
  Inventory as InventoryIcon,
} from '@mui/icons-material';

// Custom node component
const CustomNode = ({ data, selected }) => {
  const getNodeIcon = () => {
    switch (data.type) {
      case 'manufacturer':
        return <FactoryIcon fontSize="large" />;
      case 'wholesaler':
        return <InventoryIcon fontSize="large" />;
      case 'distributor':
        return <ShippingIcon fontSize="large" />;
      case 'retailer':
      default:
        return <StoreIcon fontSize="large" />;
    }
  };

  return (
    <Paper
      elevation={selected ? 8 : 2}
      sx={{
        p: 2,
        minWidth: 180,
        border: selected ? '2px solid #1976d2' : '2px solid transparent',
        borderRadius: 2,
        textAlign: 'center',
        '&:hover': {
          boxShadow: 6,
        },
      }}
    >
      <Box sx={{ color: 'primary.main' }}>{getNodeIcon()}</Box>
      <Typography variant="subtitle1">{data.label}</Typography>
      <Typography variant="body2" color="textSecondary">
        {data.type.charAt(0).toUpperCase() + data.type.slice(1)}
      </Typography>
      {data.capacity && (
        <Typography variant="caption" display="block">
          Capacity: {data.capacity}
        </Typography>
      )}
    </Paper>
  );
}

const nodeTypes = {
  custom: CustomNode,
};

const SupplyChain = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState([
    {
      id: '1',
      type: 'custom',
      data: { label: 'Manufacturer', type: 'manufacturer', capacity: '1000 units/day' },
      position: { x: 250, y: 25 },
    },
    {
      id: '2',
      type: 'custom',
      data: { label: 'Wholesaler', type: 'wholesaler', capacity: '5000 units' },
      position: { x: 100, y: 200 },
    },
    {
      id: '3',
      type: 'custom',
      data: { label: 'Distributor', type: 'distributor', capacity: '2000 units' },
      position: { x: 400, y: 200 },
    },
    {
      id: '4',
      type: 'custom',
      data: { label: 'Retailer A', type: 'retailer' },
      position: { x: 50, y: 350 },
    },
    {
      id: '5',
      type: 'custom',
      data: { label: 'Retailer B', type: 'retailer' },
      position: { x: 250, y: 350 },
    },
    {
      id: '6',
      type: 'custom',
      data: { label: 'Retailer C', type: 'retailer' },
      position: { x: 450, y: 350 },
    },
  ]);

  const [edges, setEdges, onEdgesChange] = useEdgesState([
    { id: 'e1-2', source: '1', target: '2', label: '2 days' },
    { id: 'e1-3', source: '1', target: '3', label: '3 days' },
    { id: 'e2-4', source: '2', target: '4', label: '1 day' },
    { id: 'e2-5', source: '2', target: '5', label: '1 day' },
    { id: 'e3-5', source: '3', target: '5', label: '2 days' },
    { id: 'e3-6', source: '3', target: '6', label: '2 days' },
  ]);

  const [openDialog, setOpenDialog] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeForm, setNodeForm] = useState({
    label: '',
    type: 'retailer',
    capacity: '',
  });

  const handleNodeClick = (event, node) => {
    setSelectedNode(node);
  };

  const handleAddNode = () => {
    setNodeForm({
      label: '',
      type: 'retailer',
      capacity: '',
    });
    setSelectedNode(null);
    setOpenDialog(true);
  };

  const handleEditNode = () => {
    if (selectedNode) {
      setNodeForm({
        label: selectedNode.data.label,
        type: selectedNode.data.type,
        capacity: selectedNode.data.capacity || '',
      });
      setOpenDialog(true);
    }
  };

  const handleDeleteNode = () => {
    if (selectedNode) {
      setNodes((nds) => nds.filter((node) => node.id !== selectedNode.id));
      setEdges((eds) =>
        eds.filter(
          (edge) =>
            edge.source !== selectedNode.id && edge.target !== selectedNode.id
        )
      );
      setSelectedNode(null);
    }
  };

  const handleSaveNode = () => {
    if (nodeForm.label.trim() === '') return;

    if (selectedNode) {
      // Update existing node
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === selectedNode.id) {
            return {
              ...node,
              data: {
                ...node.data,
                label: nodeForm.label,
                type: nodeForm.type,
                capacity: nodeForm.capacity,
              },
            };
          }
          return node;
        })
      );
    } else {
      // Add new node
      const newNode = {
        id: `node-${Date.now()}`,
        type: 'custom',
        data: {
          label: nodeForm.label,
          type: nodeForm.type,
          capacity: nodeForm.capacity,
        },
        position: { x: Math.random() * 400, y: Math.random() * 400 },
      };
      setNodes((nds) => [...nds, newNode]);
    }
    setOpenDialog(false);
  };

  const onConnect = useCallback(
    (params) => {
      const newEdge = {
        ...params,
        id: `e${params.source}-${params.target}`,
        label: '1 day',
      };
      setEdges((eds) => addEdge(newEdge, eds));
    },
    [setEdges]
  );

  return (
    <Box sx={{ height: 'calc(100vh - 150px)', position: 'relative' }}>
      <Box sx={{ position: 'absolute', top: 10, left: 10, zIndex: 10 }}>
        <Tooltip title="Add Node">
          <IconButton
            color="primary"
            onClick={handleAddNode}
            sx={{ backgroundColor: 'white', boxShadow: 2, mr: 1 }}
          >
            <AddIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Edit Node">
          <span>
            <IconButton
              color="primary"
              onClick={handleEditNode}
              disabled={!selectedNode}
              sx={{ backgroundColor: 'white', boxShadow: 2, mr: 1 }}
            >
              <EditIcon />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Delete Node">
          <span>
            <IconButton
              color="error"
              onClick={handleDeleteNode}
              disabled={!selectedNode}
              sx={{ backgroundColor: 'white', boxShadow: 2 }}
            >
              <DeleteIcon />
            </IconButton>
          </span>
        </Tooltip>
      </Box>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>

      <Dialog open={openDialog} onClose={() => setOpenDialog(false)}>
        <DialogTitle>
          {selectedNode ? 'Edit Node' : 'Add New Node'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2, minWidth: 300 }}>
            <TextField
              label="Node Label"
              fullWidth
              value={nodeForm.label}
              onChange={(e) =>
                setNodeForm({ ...nodeForm, label: e.target.value })
              }
              sx={{ mb: 2 }}
            />
            <TextField
              select
              label="Node Type"
              fullWidth
              value={nodeForm.type}
              onChange={(e) =>
                setNodeForm({ ...nodeForm, type: e.target.value })
              }
              sx={{ mb: 2 }}
            >
              <MenuItem value="manufacturer">Manufacturer</MenuItem>
              <MenuItem value="wholesaler">Wholesaler</MenuItem>
              <MenuItem value="distributor">Distributor</MenuItem>
              <MenuItem value="retailer">Retailer</MenuItem>
            </TextField>
            <TextField
              label="Capacity (optional)"
              fullWidth
              value={nodeForm.capacity}
              onChange={(e) =>
                setNodeForm({ ...nodeForm, capacity: e.target.value })
              }
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveNode} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SupplyChain;
