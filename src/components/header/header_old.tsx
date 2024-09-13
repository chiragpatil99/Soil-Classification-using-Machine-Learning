import React, { FC } from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import { Container } from '@mui/material';
interface HeaderProps {}

const Header: FC<HeaderProps> = () => (  
   <div >
      <AppBar className = "header" position="static">
        <Toolbar>
          <Typography variant="h6">
            Project Title X
          </Typography>
        </Toolbar>
      </AppBar>

    </div>
);

export default Header;
