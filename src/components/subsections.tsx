import * as React from 'react';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardActionArea from '@mui/material/CardActionArea';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import ImageGrid from './imagegrid';
import parse from 'html-react-parser';

interface FeaturedPostProps {
    post: {
        date?: string;
        description: string;
        title: string;
        images?: any
    };
}

export default function FeaturedPost(props: FeaturedPostProps) {
    const { post } = props;
    const paragraphs = post.description.split('\n').map((paragraph, index) => (
        <p key={index}>{parse(paragraph)}</p>
    ));
    return (
        <Grid item xs={12} md={6} >
            <Card sx={{ display: 'flex', backgroundColor: "#DFD7BF", boxShadow: "none" }}>
                <CardContent >
                    <Grid item xs={12} md={12} >
                        <Typography component="h2" variant="h5">
                            {post.title}
                        </Typography>
                        <Typography variant="subtitle1" color="text.secondary">
                            {post.date}
                        </Typography>
                        <Typography variant="subtitle1" paragraph>
                            {paragraphs}
                        </Typography>
                    </Grid>
                    <Grid item xs={12} md={12} >
                        <ImageGrid images={post.images} />
                    </Grid>
                </CardContent>
            </Card>
        </Grid>
    );
}