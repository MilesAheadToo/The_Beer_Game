import React from 'react';
import { Box } from '@chakra-ui/react';
import { Helmet } from 'react-helmet-async';

const PageLayout = ({ title, children, maxW = 'container.lg', ...rest }) => {
  return (
    <>
      <Helmet>
        <title>{title ? `${title} | Beer Game` : 'Beer Game'}</title>
        <style>
          {`
            @import url('https://fonts.googleapis.com/css2?family=Trebuchet+MS:wght@400;600;700&display=swap');
            body {
              font-family: 'Trebuchet MS', sans-serif;
            }
          `}
        </style>
      </Helmet>
      <Box 
        as="main" 
        maxW={maxW} 
        mx="auto" 
        px={4} 
        py={8}
        fontFamily="'Trebuchet MS', sans-serif"
        {...rest}
      >
        {title && (
          <Box mb={8}>
            <Box h="0.5em" /> {/* Half the height of the title */}
            <h1 style={{ 
              fontSize: '2.5rem', 
              fontWeight: 600, 
              margin: 0,
              lineHeight: 1.2,
              color: '#2D3748'
            }}>
              {title}
            </h1>
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
        color: '#2D3748'
      }}>
        {title}
      </h2>
    )}
    {children}
  </Box>
);

export default PageLayout;
