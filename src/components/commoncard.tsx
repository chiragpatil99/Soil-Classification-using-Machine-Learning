import React, { ReactNode } from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import CardActions from '@mui/material/CardActions';
import Button from '@mui/material/Button';

interface CommonCardProps {
  title: string;
  content: ReactNode;
  actionButton?: ReactNode;
}

const CommonCard: React.FC<CommonCardProps> = ({ title, content, actionButton }) => {
  return (
    <Card className='card'>
      <CardContent>
        <Typography variant="h6">{title}</Typography>
        {content}
      </CardContent>
      {actionButton && (
        <CardActions>
          {actionButton}
        </CardActions>
      )}
    </Card>
  );
};

export default CommonCard;
