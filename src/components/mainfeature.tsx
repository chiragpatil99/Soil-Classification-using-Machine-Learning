import * as React from 'react';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import ImageGrid from './imagegrid';
import CommonTable from './TableGrid';
import parse from 'html-react-parser';

interface MainFeaturedPostProps {
    post: {
        description: string;
        linkText?: string;
        title: string;
    };
    images?: any,
    childComponent?: React.ReactNode;
    tableData?: {
        data: Array<any>
        columns: Array<string>
        containerStyle: {}
        cellStyle: {}
    };
    variantStyle?: boolean


}

export default function MainFeaturedPost(props: MainFeaturedPostProps) {
    const { post, images, childComponent, tableData,variantStyle } = props;
    // variantStyle = variantStyle ? variantStyle : "h5"
    const paragraphs = post.description.split('\n').map((paragraph, index) => (
        <p key={index}>{parse(paragraph)}</p>
    ));
    console.log(images, "in main")
    return (
        <>
            <Paper className='card' style={{ borderRadius: "15px" }}
                sx={{
                    position: 'relative',
                    backgroundColor: '#DFD7BF',
                    color: '#3F2305',
                    mb: 4,
                    backgroundSize: 'cover',
                    backgroundRepeat: 'no-repeat',
                    backgroundPosition: 'center',
                }}
            >
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        right: 0,
                        left: 0,
                    }}
                />
                <Grid container>
                    <Grid >
                        <Box
                            sx={{
                                position: 'relative',
                                p: { xs: 3, md: 6 },
                                pr: { md: 0 },
                            }}
                        >
                            <Typography component="h2" variant = {variantStyle ? "h6" : "h5"} color="inherit" gutterBottom>
                                <strong>{post.title}</strong>
                            </Typography>
                            <Typography className="text-body" variant="subtitle1" color="inherit" paragraph>
                                {paragraphs}

                                {childComponent}
                            </Typography>
                        </Box>
                    </Grid>
                    <Box>
                        {tableData && <CommonTable columns={tableData.columns} data={tableData.data} containerStyle={tableData.containerStyle} cellStyle={tableData.cellStyle} />}
                    </Box>
                    <Box>
                        <ImageGrid images={images} />
                    </Box>

                </Grid>

            </Paper >
        </>
    );
}