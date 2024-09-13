import React from 'react';
import Grid from '@mui/material/Grid';
import { Paper, Typography } from '@mui/material';

interface ImageGridProps {
    images: Array<string>
    labels?: Array<string>
    stylec?: {}
}

const ImageGrid: any = (props: ImageGridProps) => {
    const { images, labels, stylec } = props;
    console.log(images, "last")

    return (
        <Grid container spacing={2} style={{ paddingTop: "10px", marginBottom: "0px", boxShadow:"none" }}>
            {labels && images?.map((image, index) => (
                
                images.length >= 2 ? <><Grid item xs={12} sm={6} md={4} lg={12/images.length} key={index} style={{ display: 'flex', objectFit: "cover" ,boxShadow:"none"}}>
                    <Paper elevation={3} sx={{ backgroundColor: 'transparent' ,boxShadow:"none"}}>
                    <img src={image} alt={`Image ${index}`} style =  { stylec ? stylec :{ maxWidth: '100%', maxHeight: '100%', width: '100%', height: '100%' }} />
                    <Typography variant="subtitle1" align="center" sx={{ backgroundColor: 'transparent', marginTop: "5px", }}>
                    <strong>{labels[index]}</strong> 
                    </Typography>
                    </Paper>
                </Grid>
                </> : <Grid item xs={12} sm={12} md={12} lg={12} key={index} style = { stylec ? stylec :{ display: 'flex', justifyContent: 'center', objectFit: "cover" }}>
                    <Paper elevation={3} sx={{ backgroundColor: 'transparent' ,boxShadow:"none"}}>
                    <Typography variant="subtitle1" align="center" sx={{ backgroundColor: 'transparent', marginTop: "5px", }}>
                    <strong>{labels[index]}</strong> 
                    </Typography>
                    <img src={image} alt={`Image ${index}`} style={{ maxWidth: '100%', maxHeight: '100%', width: '100%', height: '100%' }} />
                    </Paper>

                </Grid>

            ))}
            {/* {labels && labels?.map((label, index) => (
                labels.length > 2 ? <Grid item xs={12} sm={6} md={4} lg={12/labels.length} key={index} style={{ display: 'flex', objectFit: "cover", justifyContent: "center" }}>
                    <Typography variant="body2"><strong>{label}</strong></Typography>
                </Grid>
                    : <Grid item xs={12} sm={6} md={4} lg={3} key={index} style={{ display: 'flex', objectFit: "cover", justifyContent: "center" }}>
                        <Typography variant="body2"><strong>{label}</strong></Typography>
                    </Grid>

            ))} */}

        </Grid>
    );
};

export default ImageGrid;