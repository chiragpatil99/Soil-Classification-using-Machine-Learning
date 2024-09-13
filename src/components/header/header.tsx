import * as React from 'react';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link';
import vv from '../../static/logo_2.png';
import { Grid } from '@mui/material';

interface HeaderProps {
    sections: Array<any>;
    title: string;
}

export default function Header(props: HeaderProps) {
    const { sections, title } = props;
    console.log("sections", sections)
    return (
        <React.Fragment>
            <Toolbar sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Grid container style={{ justifyContent: "center" }}>
                    <Grid item>
                        <Typography
                            component="h2"
                            variant="h4"
                            color="#3F2305"
                            align="center"
                            noWrap
                            sx={{ flex: 1, paddingLeft: "7rem", fontWeight: "550" }}
                        >
                            {title}
                        </Typography>
                    </Grid>
                </Grid>
                <a href="https://www.vt.edu/" target="_blank" rel="noopener noreferrer">
                    <img
                        src={vv}
                        alt="VT Website Logo"
                        style={{ width: '160px', height: 'auto' }}
                    />
                </a>
            </Toolbar>
            {<Toolbar
                component="nav"
                variant="dense"
                sx={{ justifyContent: 'space-between', overflowX: 'auto' }}
            >
                <Grid container style={{ justifyContent: "space-between" }}>
                    {sections?.map((section: any) => (

                        <Grid item xs={12} sm={6} md={4} lg={3} style={{ display: "flex", justifyContent: "center" }}>
                            <Link
                                color="inherit"
                                noWrap
                                key={section.title}
                                variant="subtitle1"
                                sx={{ textAlign: "center" }}
                            >
                                {section.title}
                            </Link>
                        </Grid>

                    ))}
                </Grid>
            </Toolbar>}
        </React.Fragment>
    );
}