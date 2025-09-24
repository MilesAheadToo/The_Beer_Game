import React from 'react';
import { Box } from '@chakra-ui/react';
import { Typography } from '@mui/material';
import { Helmet } from 'react-helmet-async';

const PRIMARY_FONT = "'Trebuchet MS', 'TrebuchetMS', 'Lucida Sans Unicode', 'Lucida Grande', sans-serif";

const PageLayout = ({ title, children, maxW = 'container.lg', ...rest }) => {
  return (
    <>
      <Helmet>
        <title>{title ? `${title} | Beer Game` : 'Beer Game'}</title>
      </Helmet>
      <Box 
        as="main" 
        maxW={maxW} 
        mx="auto" 
        px={4} 
        py={8}
        fontFamily={PRIMARY_FONT}
        {...rest}
      >
        {title && (
          <Box mb={8}>
            <Box h="0.5em" />
            <Typography
              variant="h2"
              component="h1"
              color="text.primary"
              sx={{ fontWeight: 700, fontFamily: PRIMARY_FONT }}
            >
              {title}
            </Typography>
          </Box>
        )}
        {children}
      </Box>
    </>
  );
};

export const PageSection = ({ title, children, ...rest }) => (
  <Box mb={8} {...rest}>
    {title && (
      <h2 style={{
        fontSize: '1.5rem',
        fontWeight: 600,
        margin: '1.25em 0 0.75em',
        lineHeight: 1.2,
        color: '#2D3748',
        fontFamily: PRIMARY_FONT,
      }}>
        {title}
      </h2>
    )}
    {children}
  </Box>
);

export default PageLayout;
